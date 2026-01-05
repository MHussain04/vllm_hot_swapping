# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Model weight cache for hot-swapping with dynamic RAM management.

This module provides an explicit CPU RAM cache for model weights to enable
fast hot-swapping between different fine-tuned models. It implements:
- Dynamic cache size calculation based on available system RAM
- LRU eviction based on last inference time
- Automatic Linux page cache management to avoid memory duplication
"""

import glob
import os
import time
from typing import Generator

import psutil
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class ModelWeightCache:
    """
    Explicit CPU RAM cache for model weights with dynamic LRU eviction.

    This cache stores model weights in system RAM to enable fast swapping
    between models. When swapping to a cached model, weights are loaded
    from RAM (~100ms) instead of disk (~770ms).

    Features:
    - Dynamic cache size based on available system RAM
    - LRU eviction based on last_inference_time
    - Size-based eviction (evict until new model fits)
    - Automatic Linux page cache cleanup to prevent duplicates
    """

    def __init__(self, ram_utilization: float = 0.8):
        """
        Initialize the model weight cache.

        Args:
            ram_utilization: Fraction of available RAM to use for cache (default: 0.8)
        """
        self.cache = {}  # model_path → {weights, last_inference_time, size_bytes}
        self.total_size_bytes = 0
        self.ram_utilization = ram_utilization
        self.max_size_bytes = self._calculate_available_ram()

        logger.info(
            "Model weight cache initialized: max_size=%.2f GB (%.0f%% of available RAM)",
            self.max_size_bytes / (1024**3),
            ram_utilization * 100
        )

    def _calculate_available_ram(self) -> int:
        """
        Calculate available system RAM dynamically.

        Returns:
            Maximum cache size in bytes
        """
        mem = psutil.virtual_memory()
        available = int(mem.available * self.ram_utilization)

        logger.info(
            "Available RAM for cache: %.2f GB (%.2f GB total, %.2f GB available)",
            available / (1024**3),
            mem.total / (1024**3),
            mem.available / (1024**3)
        )

        return available

    def has_model(self, model_path: str) -> bool:
        """
        Check if a model is in the cache.

        Args:
            model_path: Path to the model

        Returns:
            True if model is cached, False otherwise
        """
        return model_path in self.cache

    def get_weights_iterator(
        self, model_path: str
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Get an iterator over cached model weights.

        Args:
            model_path: Path to the cached model

        Yields:
            Tuples of (parameter_name, tensor) from cache

        Raises:
            KeyError: If model is not in cache
        """
        if model_path not in self.cache:
            raise KeyError(f"Model {model_path} not found in cache")

        weights_dict = self.cache[model_path]['weights']

        logger.info(
            "Loading %d tensors from cache for %s",
            len(weights_dict),
            model_path
        )

        for name, tensor in weights_dict.items():
            yield (name, tensor)

    def cache_as_iterate(
        self,
        model_path: str,
        disk_iterator: Generator[tuple[str, torch.Tensor], None, None]
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Wrap a disk iterator to cache weights as they're loaded.

        This generator wraps the disk weight loader and caches each tensor
        in CPU RAM as it's loaded, while still yielding it for GPU loading.
        After all weights are loaded, they're added to the cache.

        Args:
            model_path: Path to the model being loaded
            disk_iterator: Original iterator from disk loader

        Yields:
            Tuples of (parameter_name, tensor) from disk
        """
        weights_dict = {}
        size_bytes = 0
        tensor_count = 0

        logger.info("Loading and caching weights for %s...", model_path)
        load_start = time.time()

        # Iterate through disk weights, caching each one
        for name, tensor in disk_iterator:
            # Clone to CPU for our cache (ensures it's on CPU)
            cached_tensor = tensor.clone().cpu()
            weights_dict[name] = cached_tensor

            # Calculate size
            tensor_size = cached_tensor.element_size() * cached_tensor.nelement()
            size_bytes += tensor_size
            tensor_count += 1

            # Yield original tensor for GPU loading
            yield (name, tensor)

        load_time = time.time() - load_start

        logger.info(
            "Loaded %d tensors (%.2f GB) from disk in %.2f seconds",
            tensor_count,
            size_bytes / (1024**3),
            load_time
        )

        # After iteration completes, add to cache
        self._add_to_cache(model_path, weights_dict, size_bytes)

        # Drop from Linux page cache to avoid duplicates
        self._drop_from_page_cache(model_path)

    def _add_to_cache(
        self,
        model_path: str,
        weights_dict: dict[str, torch.Tensor],
        size_bytes: int
    ):
        """
        Add model weights to cache with LRU eviction if needed.

        Args:
            model_path: Path to the model
            weights_dict: Dictionary of parameter names to tensors
            size_bytes: Total size of the model in bytes

        Raises:
            MemoryError: If model is too large for cache even after evicting all models
        """
        # Evict LRU models until enough space is available
        evicted_count = 0
        while self.total_size_bytes + size_bytes > self.max_size_bytes:
            if not self.cache:
                raise MemoryError(
                    f"Model too large for cache: {size_bytes / (1024**3):.2f} GB "
                    f"exceeds max cache size: {self.max_size_bytes / (1024**3):.2f} GB"
                )

            # Find model with oldest last_inference_time
            evict_model = min(
                self.cache.keys(),
                key=lambda m: self.cache[m]['last_inference_time']
            )
            evict_size = self.cache[evict_model]['size_bytes']

            logger.info(
                "Cache full - evicting %s (%.2f GB, last used %.1f min ago)",
                evict_model,
                evict_size / (1024**3),
                (time.time() - self.cache[evict_model]['last_inference_time']) / 60
            )

            del self.cache[evict_model]
            self.total_size_bytes -= evict_size
            evicted_count += 1

        # Add to cache
        self.cache[model_path] = {
            'weights': weights_dict,
            'last_inference_time': time.time(),
            'size_bytes': size_bytes
        }
        self.total_size_bytes += size_bytes

        logger.info(
            "Cached %s (%.2f GB). Cache: %.2f GB / %.2f GB (%d models%s)",
            model_path,
            size_bytes / (1024**3),
            self.total_size_bytes / (1024**3),
            self.max_size_bytes / (1024**3),
            len(self.cache),
            f", evicted {evicted_count}" if evicted_count > 0 else ""
        )

    def update_inference_time(self, model_path: str | None):
        """
        Update last_inference_time for a model (call on every inference request).

        This is used for LRU eviction - models with older timestamps are
        evicted first when cache is full.

        Args:
            model_path: Path to the currently loaded model (None is ignored)
        """
        if model_path and model_path in self.cache:
            self.cache[model_path]['last_inference_time'] = time.time()

    def _drop_from_page_cache(self, model_path: str):
        """
        Tell Linux kernel to drop model files from page cache.

        This prevents duplicate memory usage between our explicit cache
        and Linux's automatic page cache.

        Args:
            model_path: Path to the model directory
        """
        try:
            dropped_count = 0
            for file in glob.glob(f"{model_path}/*.safetensors"):
                try:
                    fd = os.open(file, os.O_RDONLY)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    os.close(fd)
                    dropped_count += 1
                except OSError as e:
                    logger.warning(
                        "Could not drop %s from page cache: %s",
                        file, e
                    )

            if dropped_count > 0:
                logger.info(
                    "Dropped %d files from Linux page cache for %s",
                    dropped_count,
                    model_path
                )
        except Exception as e:
            logger.warning(
                "Error dropping %s from page cache: %s",
                model_path, e
            )

    def get_cache_stats(self) -> dict:
        """
        Get current cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            'num_models': len(self.cache),
            'total_size_gb': self.total_size_bytes / (1024**3),
            'max_size_gb': self.max_size_bytes / (1024**3),
            'utilization_percent': (self.total_size_bytes / self.max_size_bytes * 100)
                                   if self.max_size_bytes > 0 else 0,
            'cached_models': list(self.cache.keys())
        }


# Global cache instance
model_weight_cache = ModelWeightCache()
