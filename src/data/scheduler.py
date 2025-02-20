"""
Scheduler for running data ingestion jobs at specified intervals.
"""

import schedule
import time
from typing import List, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestionScheduler:
    def __init__(self):
        self.jobs: List[Callable] = []

    def add_job(self, job: Callable, interval_minutes: int):
        """Add a job to run at specified interval"""
        schedule.every(interval_minutes).minutes.do(self._run_job, job)
        self.jobs.append(job)
        logger.info(f"Added job {job.__name__} to run every {interval_minutes} minutes")

    def _run_job(self, job: Callable):
        """Run a job and handle any errors"""
        try:
            logger.info(f"Running job: {job.__name__}")
            job()
            logger.info(f"Completed job: {job.__name__}")
        except Exception as e:
            logger.error(f"Error in job {job.__name__}: {str(e)}")

    def start(self):
        """Start the scheduler"""
        logger.info("Starting scheduler...")
        while True:
            schedule.run_pending()
            time.sleep(60) 