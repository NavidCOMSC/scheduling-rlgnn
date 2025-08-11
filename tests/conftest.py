import pytest
from job_shop_lib import JobShopInstance, Operation
from scheduling_rlgnn.envwrapper import JobShopEnvironmentWrapper


@pytest.fixture(name="minimal_job_shop_instance")
def minimal_job_shop_instance_fixture():
    """Creates a minimal JobShopInstance with 2 jobs and 2 machines."""
    jobs = [
        [Operation(0, duration=3), Operation(1, duration=2)],
        [Operation(1, duration=2), Operation(0, duration=4)],
    ]
    return JobShopInstance(jobs=jobs, num_machines=2, name="MinimalInstance")
