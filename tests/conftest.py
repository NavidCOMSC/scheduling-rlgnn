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


@pytest.fixture(name="job_shop_environment_wrapper")
def job_shop_environment_wrapper_fixture(minimal_job_shop_instance):
    """Provides a JobShopEnvironmentWrapper instance for testing."""
    return JobShopEnvironmentWrapper(minimal_job_shop_instance)


@pytest.fixture(name="initialized_environment")
def initialized_environment_fixture(job_shop_environment_wrapper):
    """Provides a pre-initialized environment (after reset)."""
    job_shop_environment_wrapper.reset()
    return job_shop_environment_wrapper


@pytest.fixture(name="environment_after_step")
def environment_after_step_fixture(initialized_environment):
    """Provides an environment after one valid step has been taken."""
    # Take a valid action (index 0)
    initialized_environment.step(0)
    return initialized_environment


@pytest.fixture(name="environment_with_completed_operation")
def environment_with_completed_operation_fixture(initialized_environment):
    """Provides an environment where one operation has been completed."""
    # Execute a valid action
    initialized_environment._execute_action(0)
    return initialized_environment
