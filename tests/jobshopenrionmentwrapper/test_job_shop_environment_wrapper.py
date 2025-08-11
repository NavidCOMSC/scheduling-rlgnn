import pytest
import numpy as np
from job_shop_lib import JobShopInstance, Operation
from scheduling_rlgnn.envwrapper import JobShopEnvironmentWrapper


# Fixture for a minimal JobShopInstance
@pytest.fixture
def minimal_job_shop_instance():
    """Creates a minimal JobShopInstance with 2 jobs and 2 machines."""
    jobs = [
        [Operation(0, duration=3), Operation(1, duration=2)],
        [Operation(1, duration=2), Operation(0, duration=4)],
    ]
    return JobShopInstance(jobs=jobs, num_machines=2)
