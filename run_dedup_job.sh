#!/bin/bash
cd ~/rain
. venv/bin/activate
python -m src.scripts.deduplication_job >> logs/dedup_job.log 2>&1
