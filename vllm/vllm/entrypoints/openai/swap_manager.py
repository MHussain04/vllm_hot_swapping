# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Manager for handling asynchronous model weight swap operations.

This module provides the SwapManager class which handles async model swap jobs,
including HuggingFace model downloads and tracking job status.
"""

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.engine.protocol import EngineClient
    from vllm.entrypoints.openai.serving_models import OpenAIServingModels

logger = init_logger(__name__)


class SwapJobStatus(str, Enum):
    """Status of a model swap job."""

    PENDING = "pending"
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    SWAPPING = "swapping"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SwapJob:
    """Represents a model weight swap job."""

    job_id: str
    new_model_path: str
    new_revision: str | None
    swap_tokenizer: bool
    status: SwapJobStatus
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    local_model_path: str | None = None  # Path after download (if HF model)


class SwapManager:
    """Manages asynchronous model weight swap operations.

    This class handles:
    - Tracking swap job status
    - Async HuggingFace model downloads
    - Coordinating with the engine to perform swaps
    - Ensuring only one swap happens at a time
    """

    def __init__(
        self,
        engine: "EngineClient",
        download_dir: str | None = None,
        serving_models: "OpenAIServingModels | None" = None,
    ):
        """Initialize the SwapManager.

        Args:
            engine: The vLLM engine client
            download_dir: Directory to download HuggingFace models to
            serving_models: OpenAIServingModels instance for updating model registry
        """
        self.engine = engine
        self.download_dir = download_dir
        self.serving_models = serving_models
        self.jobs: dict[str, SwapJob] = {}
        self._current_job_id: str | None = None
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _is_local_path(self, model_path: str) -> bool:
        """Check if the model path is a local directory."""
        return os.path.isdir(model_path)

    def _download_from_hf(
        self,
        model_id: str,
        revision: str | None,
    ) -> str:
        """Download model from HuggingFace Hub (runs in thread pool).

        Args:
            model_id: HuggingFace model ID
            revision: Model revision

        Returns:
            Local path to downloaded model
        """
        logger.info("Downloading model from HuggingFace: %s", model_id)
        local_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=self.download_dir,
            local_dir_use_symlinks=False,
        )
        logger.info("Download completed: %s -> %s", model_id, local_path)
        return local_path

    async def _execute_swap(self, job: SwapJob) -> None:
        """Execute the actual weight swap operation.

        Args:
            job: The swap job to execute
        """
        try:
            # Step 1: Download if HuggingFace model
            if not self._is_local_path(job.new_model_path):
                job.status = SwapJobStatus.DOWNLOADING
                loop = asyncio.get_running_loop()
                job.local_model_path = await loop.run_in_executor(
                    self._executor,
                    self._download_from_hf,
                    job.new_model_path,
                    job.new_revision,
                )
            else:
                job.local_model_path = job.new_model_path

            # Step 2: Validate and swap
            job.status = SwapJobStatus.VALIDATING
            model_path_to_use = job.local_model_path or job.new_model_path

            job.status = SwapJobStatus.SWAPPING
            await self.engine.swap_weights(
                new_model_path=model_path_to_use,
                new_revision=job.new_revision if job.local_model_path is None else None,
            )

            # Update model registry - remove old model names and add new one
            if self.serving_models is not None:
                from vllm.entrypoints.openai.serving_models import BaseModelPath

                # Save old model names for logging
                old_models = [m.name for m in self.serving_models.base_model_paths]

                # Remove all old model names (they pointed to old weights)
                self.serving_models.base_model_paths.clear()

                # Add the new model name
                self.serving_models.base_model_paths.append(
                    BaseModelPath(name=job.new_model_path, model_path=job.new_model_path)
                )

                logger.info(
                    "Model registry updated: removed %s, added '%s'",
                    old_models,
                    job.new_model_path,
                )

            # Step 3: Mark complete
            job.status = SwapJobStatus.COMPLETED
            job.completed_at = datetime.now()
            logger.info("Swap job %s completed successfully", job.job_id)

        except Exception as e:
            job.status = SwapJobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.exception("Swap job %s failed: %s", job.job_id, e)
            raise

        finally:
            async with self._lock:
                self._current_job_id = None

    async def create_swap_job(
        self,
        new_model_path: str,
        new_revision: str | None = None,
        swap_tokenizer: bool = False,
    ) -> SwapJob:
        """Create a new swap job.

        Args:
            new_model_path: Path to the new model (local or HuggingFace ID)
            new_revision: Optional revision for HuggingFace models
            swap_tokenizer: Whether to also swap the tokenizer

        Returns:
            The created SwapJob

        Raises:
            ValueError: If a swap is already in progress
        """
        async with self._lock:
            if self._current_job_id is not None:
                current_job = self.jobs.get(self._current_job_id)
                if current_job and current_job.status not in (
                    SwapJobStatus.COMPLETED,
                    SwapJobStatus.FAILED,
                ):
                    raise ValueError(
                        f"A swap job is already in progress: {self._current_job_id}"
                    )

            job_id = str(uuid.uuid4())
            job = SwapJob(
                job_id=job_id,
                new_model_path=new_model_path,
                new_revision=new_revision,
                swap_tokenizer=swap_tokenizer,
                status=SwapJobStatus.PENDING,
            )
            self.jobs[job_id] = job
            self._current_job_id = job_id

        # Start the swap in background
        job.started_at = datetime.now()
        asyncio.create_task(self._execute_swap(job))

        return job

    def get_job(self, job_id: str) -> SwapJob | None:
        """Get a swap job by ID.

        Args:
            job_id: The job ID

        Returns:
            The SwapJob if found, None otherwise
        """
        return self.jobs.get(job_id)

    def get_current_job(self) -> SwapJob | None:
        """Get the currently running swap job.

        Returns:
            The current SwapJob if any, None otherwise
        """
        if self._current_job_id:
            return self.jobs.get(self._current_job_id)
        return None

    def list_jobs(self) -> list[SwapJob]:
        """List all swap jobs.

        Returns:
            List of all SwapJobs
        """
        return list(self.jobs.values())

    def cleanup_old_jobs(self, max_jobs: int = 100) -> None:
        """Remove old completed/failed jobs if we have too many.

        Args:
            max_jobs: Maximum number of jobs to keep
        """
        if len(self.jobs) <= max_jobs:
            return

        # Sort by created_at and remove oldest completed/failed jobs
        completed_jobs = [
            j
            for j in self.jobs.values()
            if j.status in (SwapJobStatus.COMPLETED, SwapJobStatus.FAILED)
        ]
        completed_jobs.sort(key=lambda j: j.created_at)

        jobs_to_remove = len(self.jobs) - max_jobs
        for job in completed_jobs[:jobs_to_remove]:
            del self.jobs[job.job_id]
