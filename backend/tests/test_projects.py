import pytest
from pydantic import ValidationError

from app.api.projects import ProjectCreate, ProjectUpdate


@pytest.mark.parametrize("name", ["", "   ", "\t\n"])
def test_project_names_cannot_be_blank(name):
    with pytest.raises(ValidationError):
        ProjectCreate(name=name)
    with pytest.raises(ValidationError):
        ProjectUpdate(name=name)


def test_project_names_are_normalized():
    assert ProjectCreate(name="  My project  ").name == "My project"
