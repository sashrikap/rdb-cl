from rdb.exps.active_user import ExperimentActiveUser, run_experiment_server
import pytest

## Test user apis
experiment = run_experiment_server(test_mode=True)
# experiment.run()


def dict_equal(da, db):
    for key, val in da.items():
        if key not in db:
            return False
        if db[key] != val:
            return False
    return True


@pytest.mark.parametrize("joint_mode", [True, False])
def test_start(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    server_state = experiment.start_design(user_id)
    assert server_state["user_id"] == user_id
    assert server_state["curr_weights"] == None
    assert server_state["curr_trial"] == -1
    assert server_state["curr_method_idx"] == -1
    assert server_state["curr_method"] == None
    assert server_state["is_training"] == True
    assert server_state["proposal_iter"] == -1
    assert server_state["final_training_weights"] == []
    assert server_state["curr_mp4s"] == []

    training_tasks = server_state["all_training_tasks"]
    training_imgs = server_state["all_training_imgs"]
    assert len(training_tasks) > 0
    assert len(training_tasks) == len(training_imgs)
    assert server_state["training_iter"] == 0
    assert len(server_state["all_training_weights"]) == len(training_tasks)

    if joint_mode:
        assert len(server_state["curr_tasks"]) == len(training_tasks)
        assert len(server_state["curr_imgs"]) == len(training_imgs)
    else:
        assert len(server_state["curr_tasks"]) == 1
        assert len(server_state["curr_imgs"]) == 1

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert server_state["all_proposal_weights"][key] == []
        assert server_state["all_proposal_tasks"][key] == []
        assert server_state["all_proposal_imgs"][key] == []
        assert server_state["final_proposal_weights"][key] == []

    assert server_state["need_new_proposal"] == False


@pytest.mark.parametrize("joint_mode", [True, False])
def test_try_design(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)

    server_state = experiment.try_design(user_id, weights, user_state)
    assert server_state["user_id"] == user_id
    assert dict_equal(server_state["curr_weights"], weights)
    assert server_state["curr_trial"] == 0
    assert server_state["curr_method_idx"] == -1
    assert server_state["curr_method"] == None
    assert server_state["is_training"] == True
    assert server_state["proposal_iter"] == -1
    assert server_state["final_training_weights"] == []

    training_tasks = server_state["all_training_tasks"]
    training_imgs = server_state["all_training_imgs"]
    assert len(training_tasks) > 0
    assert len(training_tasks) == len(training_imgs)
    assert server_state["training_iter"] == 0
    assert len(server_state["all_training_weights"]) == len(training_tasks)
    assert len(server_state["all_training_weights"][0]) == 1

    if joint_mode:
        assert len(server_state["curr_tasks"]) == len(training_tasks)
        assert len(server_state["curr_imgs"]) == len(training_imgs)
        assert len(server_state["curr_mp4s"]) == len(training_tasks)
    else:
        assert len(server_state["curr_tasks"]) == 1
        assert len(server_state["curr_imgs"]) == 1
        assert len(server_state["curr_mp4s"]) == 1

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert server_state["all_proposal_weights"][key] == []
        assert server_state["all_proposal_tasks"][key] == []
        assert server_state["all_proposal_imgs"][key] == []
        assert server_state["final_proposal_weights"][key] == []

    assert server_state["need_new_proposal"] == False


@pytest.mark.parametrize("joint_mode", [True, False])
def test_make_video(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    server_state = experiment.try_design(user_id, weights, user_state)

    num_videos = len(server_state["curr_mp4s"])
    for idx in range(num_videos):
        user_state = server_state

        url = experiment.make_video(user_id, idx, user_state)

        assert url is not None and type(url) == str
        assert server_state["user_id"] == user_id
        assert dict_equal(server_state["curr_weights"], weights)
        assert server_state["curr_trial"] == 0
        assert server_state["curr_method_idx"] == -1
        assert server_state["curr_method"] == None
        assert server_state["is_training"] == True
        assert server_state["proposal_iter"] == -1
        assert server_state["final_training_weights"] == []

        training_tasks = server_state["all_training_tasks"]
        training_imgs = server_state["all_training_imgs"]
        assert len(training_tasks) > 0
        assert len(training_tasks) == len(training_imgs)
        assert server_state["training_iter"] == 0
        assert len(server_state["all_training_weights"]) == len(training_tasks)
        assert len(server_state["all_training_weights"][0]) == 1
        if joint_mode:
            assert len(server_state["curr_tasks"]) == len(training_tasks)
            assert len(server_state["curr_imgs"]) == len(training_imgs)
            assert len(server_state["curr_mp4s"]) == len(training_tasks)
        else:
            assert len(server_state["curr_tasks"]) == 1
            assert len(server_state["curr_imgs"]) == 1
            assert len(server_state["curr_mp4s"]) == 1

        curr_active_keys = server_state["curr_active_keys"]
        for key in curr_active_keys:
            assert server_state["all_proposal_weights"][key] == []
            assert server_state["all_proposal_tasks"][key] == []
            assert server_state["all_proposal_imgs"][key] == []
            assert server_state["final_proposal_weights"][key] == []

        assert server_state["need_new_proposal"] == False


@pytest.mark.parametrize("joint_mode", [True, False])
def test_second_try(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    server_state = experiment.try_design(user_id, weights, user_state)

    num_videos = len(server_state["curr_mp4s"])
    for idx in range(num_videos):
        user_state = server_state

        url = experiment.make_video(user_id, idx, user_state)

        assert url is not None and type(url) == str
        assert server_state["user_id"] == user_id
        assert dict_equal(server_state["curr_weights"], weights)
        assert server_state["curr_trial"] == 1  ## Added 1
        assert server_state["curr_method_idx"] == -1
        assert server_state["curr_method"] == None
        assert server_state["is_training"] == True
        assert server_state["training_iter"] == 0
        assert server_state["proposal_iter"] == -1
        assert server_state["final_training_weights"] == []

        training_tasks = server_state["all_training_tasks"]
        training_imgs = server_state["all_training_imgs"]
        assert len(training_tasks) > 0
        assert len(training_tasks) == len(training_imgs)
        assert server_state["training_iter"] == 0
        assert len(server_state["all_training_weights"]) == len(training_tasks)
        assert len(server_state["all_training_weights"][0]) == 2  # Added 1
        if joint_mode:
            assert len(server_state["curr_tasks"]) == len(training_tasks)
            assert len(server_state["curr_imgs"]) == len(training_imgs)
            assert len(server_state["curr_mp4s"]) == len(training_tasks)
        else:
            assert len(server_state["curr_tasks"]) == 1
            assert len(server_state["curr_imgs"]) == 1
            assert len(server_state["curr_mp4s"]) == 1

        curr_active_keys = server_state["curr_active_keys"]
        for key in curr_active_keys:
            assert server_state["all_proposal_weights"][key] == []
            assert server_state["all_proposal_tasks"][key] == []
            assert server_state["all_proposal_imgs"][key] == []
            assert server_state["final_proposal_weights"][key] == []

        assert server_state["need_new_proposal"] == False


@pytest.mark.parametrize("joint_mode", [True, False])
def test_accept_training_design(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    server_state = user_state
    num_videos = len(server_state["curr_mp4s"])

    assert server_state["user_id"] == user_id
    assert server_state["curr_weights"] == None  # refreshed
    assert server_state["curr_trial"] == -1  # refreshed
    assert server_state["curr_method_idx"] == -1
    assert server_state["curr_method"] == None
    assert server_state["proposal_iter"] == -1

    training_tasks = server_state["all_training_tasks"]
    training_imgs = server_state["all_training_imgs"]
    assert len(server_state["all_training_weights"]) == len(training_tasks)
    assert len(server_state["all_training_weights"][0]) == 2
    if joint_mode:
        assert server_state["is_training"] == False  # changed
        assert len(server_state["final_training_weights"]) == len(training_tasks)
        assert server_state["training_iter"] == 1
        assert len(server_state["curr_tasks"]) == 0  # refreshed
        assert len(server_state["curr_imgs"]) == 0  # refreshed
        assert len(server_state["curr_mp4s"]) == 0  # refreshed

    else:
        assert server_state["is_training"] == True  # changed
        assert len(server_state["final_training_weights"]) == 1
        assert server_state["training_iter"] == 1
        assert len(server_state["curr_tasks"]) == 1  # refreshed
        assert len(server_state["curr_imgs"]) == 1  # refreshed
        assert len(server_state["curr_mp4s"]) == 0  # refreshed

    assert len(training_tasks) > 0
    assert len(training_tasks) == len(training_imgs)

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert server_state["all_proposal_weights"][key] == []
        assert server_state["all_proposal_tasks"][key] == []
        assert server_state["all_proposal_imgs"][key] == []
        assert server_state["final_proposal_weights"][key] == []

    if joint_mode:
        assert server_state["need_new_proposal"] == True
    else:
        assert server_state["need_new_proposal"] == False


@pytest.mark.parametrize("joint_mode", [True, False])
def test_finish_training_design(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    while not user_state["need_new_proposal"]:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)

    server_state = user_state
    num_videos = len(server_state["curr_mp4s"])

    assert server_state["user_id"] == user_id
    assert server_state["curr_weights"] == None  # refreshed
    assert server_state["curr_trial"] == -1  # refreshed
    assert server_state["curr_method_idx"] == -1
    assert server_state["curr_method"] == None
    assert server_state["proposal_iter"] == -1

    training_tasks = server_state["all_training_tasks"]
    training_imgs = server_state["all_training_imgs"]
    assert len(server_state["all_training_weights"]) == len(training_tasks)
    assert len(server_state["all_training_weights"][0]) == 2
    assert server_state["is_training"] == False  # changed
    assert len(server_state["final_training_weights"]) == len(training_tasks)
    assert len(server_state["curr_tasks"]) == 0  # refreshed
    assert len(server_state["curr_imgs"]) == 0  # refreshed
    assert len(server_state["curr_mp4s"]) == 0  # refreshed
    if joint_mode:
        assert server_state["training_iter"] == 1
    else:
        assert server_state["training_iter"] == len(training_tasks)

    assert len(training_tasks) > 0
    assert len(training_tasks) == len(training_imgs)

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert server_state["all_proposal_weights"][key] == []
        assert server_state["all_proposal_tasks"][key] == []
        assert server_state["all_proposal_imgs"][key] == []
        assert server_state["final_proposal_weights"][key] == []

    assert server_state["need_new_proposal"] == True


@pytest.mark.parametrize("joint_mode", [True, False])
def test_one_proposal_design(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    while not user_state["need_new_proposal"]:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)
    server_state = experiment.new_proposal(user_id, user_state)

    assert server_state["user_id"] == user_id
    assert server_state["curr_trial"] == -1
    assert server_state["curr_method_idx"] == 0
    assert server_state["curr_method"] != None
    assert server_state["proposal_iter"] == 0

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert len(server_state["all_proposal_weights"][key]) == 1
        assert len(server_state["all_proposal_weights"][key][0]) == 0
        assert len(server_state["all_proposal_tasks"][key]) == 1
        assert len(server_state["all_proposal_imgs"][key]) == 1
        assert server_state["is_training"] == False
        assert server_state["proposal_iter"] == 0

    training_tasks = server_state["all_training_tasks"]
    training_imgs = server_state["all_training_imgs"]
    if joint_mode:
        assert len(server_state["curr_tasks"]) == 1 + len(training_tasks)
        assert len(server_state["curr_imgs"]) == 1 + len(training_imgs)
    else:
        assert len(server_state["curr_tasks"]) == 1
        assert len(server_state["curr_imgs"]) == 1

    assert server_state["need_new_proposal"] == False


@pytest.mark.parametrize("joint_mode", [True, False])
def test_one_proposal_try(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    while not user_state["need_new_proposal"]:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)
    user_state = experiment.new_proposal(user_id, user_state)
    server_state = experiment.try_design(user_id, weights, user_state)

    assert server_state["user_id"] == user_id
    assert dict_equal(server_state["curr_weights"], weights)
    assert server_state["curr_trial"] == 0
    assert server_state["curr_method_idx"] == 0
    assert server_state["curr_method"] != None
    assert server_state["proposal_iter"] == 0

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert len(server_state["all_proposal_weights"][key]) == 1
        if key == server_state["curr_method"]:
            assert len(server_state["all_proposal_weights"][key][0]) == 1  # changed
        assert len(server_state["all_proposal_tasks"][key]) == 1
        assert len(server_state["all_proposal_imgs"][key]) == 1
        assert server_state["is_training"] == False
        assert server_state["proposal_iter"] == 0

    training_tasks = server_state["all_training_tasks"]
    training_imgs = server_state["all_training_imgs"]
    if joint_mode:
        assert len(server_state["curr_tasks"]) == 1 + len(training_tasks)
        assert len(server_state["curr_imgs"]) == 1 + len(training_imgs)
    else:
        assert len(server_state["curr_tasks"]) == 1
        assert len(server_state["curr_imgs"]) == 1
    assert server_state["need_new_proposal"] == False


@pytest.mark.parametrize("joint_mode", [True, False])
def test_one_proposal_mp4(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    while not user_state["need_new_proposal"]:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)
    user_state = experiment.new_proposal(user_id, user_state)
    user_state = experiment.try_design(user_id, weights, user_state)
    server_state = user_state

    num_videos = len(server_state["curr_mp4s"])
    assert server_state["curr_trial"] == 0
    for idx in range(num_videos):
        user_state = server_state
        url = experiment.make_video(user_id, idx, user_state)
        assert "training" not in url
        assert "proposal" in url

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert len(server_state["all_proposal_weights"][key]) == 1
        assert len(server_state["all_proposal_tasks"][key]) == 1
        assert len(server_state["all_proposal_imgs"][key]) == 1
        assert server_state["final_proposal_weights"][key] == []

    assert server_state["need_new_proposal"] == False


@pytest.mark.parametrize("joint_mode", [True, False])
def test_one_proposal_accept(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    while not user_state["need_new_proposal"]:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)
    user_state = experiment.new_proposal(user_id, user_state)

    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.try_design(user_id, weights, user_state)
    server_state = experiment.next_design(user_id, user_state)

    assert server_state["user_id"] == user_id
    assert server_state["curr_trial"] == -1
    assert server_state["curr_method_idx"] == 1
    assert server_state["curr_method"] != None
    assert server_state["proposal_iter"] == 0

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert len(server_state["all_proposal_tasks"][key]) == 1
        assert len(server_state["all_proposal_weights"][key]) == 1
        if key == curr_active_keys[0]:
            assert len(server_state["all_proposal_weights"][key][0]) == 2
            assert len(server_state["final_proposal_weights"][key]) == 1
        else:
            assert len(server_state["final_proposal_weights"][key]) == 0
        assert len(server_state["all_proposal_imgs"][key]) == 1
        assert server_state["is_training"] == False  # changed
        assert server_state["proposal_iter"] == 0

    assert server_state["need_new_proposal"] == False


@pytest.mark.parametrize("joint_mode", [True, False])
def test_two_proposal_try(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    while not user_state["need_new_proposal"]:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)
    user_state = experiment.new_proposal(user_id, user_state)

    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    server_state = experiment.try_design(user_id, weights, user_state)

    assert server_state["user_id"] == user_id
    assert dict_equal(server_state["curr_weights"], weights)
    assert server_state["curr_trial"] == 0
    assert server_state["curr_method_idx"] == 1
    assert server_state["curr_method"] != None
    assert server_state["proposal_iter"] == 0

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert len(server_state["all_proposal_weights"][key]) == 1
        if key in curr_active_keys[:2]:
            assert len(server_state["all_proposal_weights"][key][0]) == 1
        if key in curr_active_keys[:1]:
            assert len(server_state["final_proposal_weights"][key]) == 1
        else:
            assert len(server_state["final_proposal_weights"][key]) == 0
        assert len(server_state["all_proposal_tasks"][key]) == 1
        assert len(server_state["all_proposal_imgs"][key]) == 1
        assert server_state["is_training"] == False  # changed
        assert server_state["proposal_iter"] == 0

    training_tasks = server_state["all_training_tasks"]
    training_imgs = server_state["all_training_imgs"]
    if joint_mode:
        assert len(server_state["curr_tasks"]) == 1 + len(training_tasks)
        assert len(server_state["curr_imgs"]) == 1 + len(training_imgs)
    else:
        assert len(server_state["curr_tasks"]) == 1
        assert len(server_state["curr_imgs"]) == 1


@pytest.mark.parametrize("joint_mode", [True, False])
def test_two_proposal_accept(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    while not user_state["need_new_proposal"]:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)
    user_state = experiment.new_proposal(user_id, user_state)

    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    user_state = experiment.try_design(user_id, weights, user_state)
    server_state = experiment.next_design(user_id, user_state)

    assert server_state["user_id"] == user_id
    assert server_state["curr_trial"] == -1
    assert server_state["curr_method_idx"] == 2
    assert server_state["curr_method"] != None
    assert server_state["proposal_iter"] == 0

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert len(server_state["all_proposal_weights"][key]) == 1
        assert len(server_state["all_proposal_tasks"][key]) == 1
        assert len(server_state["all_proposal_imgs"][key]) == 1
        if key in curr_active_keys[:2]:
            assert len(server_state["all_proposal_weights"][key][0]) == 1
        if key in curr_active_keys[:2]:
            assert len(server_state["final_proposal_weights"][key]) == 1
        else:
            assert len(server_state["final_proposal_weights"][key]) == 0
        assert server_state["is_training"] == False  # changed
        assert server_state["proposal_iter"] == 0
    training_tasks = server_state["all_training_tasks"]
    training_imgs = server_state["all_training_imgs"]
    if joint_mode:
        assert len(server_state["curr_tasks"]) == 1 + len(training_tasks)
        assert len(server_state["curr_imgs"]) == 1 + len(training_imgs)
    else:
        assert len(server_state["curr_tasks"]) == 1
        assert len(server_state["curr_imgs"]) == 1


@pytest.mark.parametrize("joint_mode", [True, False])
def test_two_proposal_iteration(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    while not user_state["need_new_proposal"]:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)
    server_state = experiment.new_proposal(user_id, user_state)

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)
    server_state = user_state

    assert server_state["need_new_proposal"] == True
    assert server_state["user_id"] == user_id
    assert server_state["curr_weights"] == None
    assert server_state["curr_trial"] == -1
    assert server_state["curr_method_idx"] == -1
    assert server_state["curr_method"] == None
    assert server_state["proposal_iter"] == 0

    for key in curr_active_keys:
        assert len(server_state["all_proposal_weights"][key]) == 1
        assert len(server_state["all_proposal_tasks"][key]) == 1
        assert len(server_state["all_proposal_imgs"][key]) == 1
        assert server_state["is_training"] == False  # changed
        assert server_state["proposal_iter"] == 0


@pytest.mark.parametrize("joint_mode", [True, False])
def test_two_proposal_iteration_start(joint_mode):
    experiment._joint_mode = joint_mode

    user_id = "abc"
    weights = {"dist": 1, "car": 2}
    user_state = experiment.start_design(user_id)
    user_state = experiment.try_design(user_id, weights, user_state)
    user_state = experiment.next_design(user_id, user_state)
    while not user_state["need_new_proposal"]:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)
    server_state = experiment.new_proposal(user_id, user_state)

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        user_state = experiment.try_design(user_id, weights, user_state)
        user_state = experiment.next_design(user_id, user_state)

    user_state = experiment.new_proposal(user_id, user_state)
    server_state = user_state

    assert server_state["need_new_proposal"] == False
    assert server_state["user_id"] == user_id
    assert server_state["curr_weights"] == None
    assert server_state["curr_trial"] == -1
    assert server_state["curr_method_idx"] == 0
    assert server_state["curr_method"] != None
    assert server_state["proposal_iter"] == 1

    curr_active_keys = server_state["curr_active_keys"]
    for key in curr_active_keys:
        assert len(server_state["all_proposal_weights"][key]) == 2
        assert len(server_state["all_proposal_tasks"][key]) == 2
        assert len(server_state["all_proposal_imgs"][key]) == 2
        assert server_state["is_training"] == False  # changed
        assert server_state["proposal_iter"] == 1
