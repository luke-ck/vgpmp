from pathlib import Path
import pybullet_data
import pytest
from gpflow_vgpmp.utils.bullet_object import BaseObject


def get_cube():
    return BaseObject(name="cube", path=None)


def get_duck():
    return BaseObject(name="duck", path=None)


def get_table():
    return BaseObject(name="table", path=None)


def get_scene_with_objects(scene):
    obj1 = get_cube()
    obj2 = get_duck()
    obj3 = get_table()

    scene.objects.append(obj1)
    scene.objects.append(obj2)
    scene.objects.append(obj3)

    return scene, obj1, obj2, obj3


def test_add_object(mock_bare_scene):
    position = [1, 2, 3]
    orientation = [0, 0, 0, 1]
    pybullet_data_path = pybullet_data.getDataPath()
    pybullet_data_path = Path(pybullet_data_path)

    random_object_path = pybullet_data_path / "cube.urdf"

    mock_bare_scene.add_object("plane", None, position, orientation)

    assert len(mock_bare_scene.objects) == 1
    assert mock_bare_scene.objects[0].name == "plane"
    assert mock_bare_scene.objects[0].position == position
    assert mock_bare_scene.objects[0].orientation == orientation
    assert mock_bare_scene.objects[0].path == pybullet_data_path / "plane_transparent.urdf"

    mock_bare_scene.add_object("plane", random_object_path, None, None)

    assert len(mock_bare_scene.objects) == 2
    assert mock_bare_scene.objects[0].name == "cube"
    assert mock_bare_scene.objects[0].position == [0, 0, 0]
    assert mock_bare_scene.objects[0].orientation == [0, 0, 0, 1]
    assert mock_bare_scene.objects[0].path == random_object_path


def test_remove_object_with_index(mock_bare_scene):
    scene, obj1, obj2, obj3 = get_scene_with_objects(mock_bare_scene)
    mock_bare_scene.remove_object(obj2, index=1)
    assert len(mock_bare_scene.objects) == 2
    assert mock_bare_scene.objects[0] == obj1
    assert mock_bare_scene.objects[1] == obj3

    mock_bare_scene.remove_object(obj1, index=0)
    assert len(mock_bare_scene.objects) == 1
    assert mock_bare_scene.objects[0] == obj3


def test_remove_object_without_index(mock_bare_scene):
    scene, obj1, obj2, obj3 = get_scene_with_objects(mock_bare_scene)

    mock_bare_scene.remove_object(obj2)
    assert len(mock_bare_scene.objects) == 2
    assert mock_bare_scene.objects[0] == obj1
    assert mock_bare_scene.objects[1] == obj3

    mock_bare_scene.remove_object(obj1)
    assert len(mock_bare_scene.objects) == 1
    assert mock_bare_scene.objects[0] == obj3


def test_remove_object_by_name(mock_bare_scene):
    scene, obj1, obj2, obj3 = get_scene_with_objects(mock_bare_scene)

    mock_bare_scene.remove_object_by_name("duck")
    assert len(mock_bare_scene.objects) == 2
    assert mock_bare_scene.objects[0] == obj1
    assert mock_bare_scene.objects[1] == obj3

    mock_bare_scene.remove_object_by_name("cube")
    assert len(mock_bare_scene.objects) == 1
    assert mock_bare_scene.objects[0] == obj3

    with pytest.raises(ValueError):
        mock_bare_scene.remove_object_by_name("NonexistentObject")


def test_get_object_by_name(mock_bare_scene):
    scene, obj1, obj2, obj3 = get_scene_with_objects(mock_bare_scene)

    obj = mock_bare_scene.get_object_by_name("duck")
    assert obj == obj2

    obj = mock_bare_scene.get_object_by_name("cube")
    assert obj == obj1

    obj = mock_bare_scene.get_object_by_name("NonexistentObject")
    assert obj is None


def test_get_object_by_index(mock_bare_scene):
    scene, obj1, obj2, obj3 = get_scene_with_objects(mock_bare_scene)

    obj = mock_bare_scene.get_object_by_index(1)
    assert obj == obj2

    obj = mock_bare_scene.get_object_by_index(0)
    assert obj == obj1

    obj = mock_bare_scene.get_object_by_index(3)
    assert obj is None


def test_get_index_by_name(mock_bare_scene):
    scene, obj1, obj2, obj3 = get_scene_with_objects(mock_bare_scene)

    index = scene.get_index_by_name("cube")
    assert index == 0

    index = scene.get_index_by_name("duck")
    assert index == 1

    index = scene.get_index_by_name("NonexistentObject")
    assert index is None

    # add another cube
    obj4 = get_cube()

    scene.objects.append(obj4)

    index = scene.get_index_by_name("cube")
    assert index == 0
