import inspect
import traceback

import pytest
from skbase.testing import BaseFixtureGenerator as _BaseFixtureGenerator, QuickTester
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.base.base_object import all_objects


class BaseFixtureGenerator(_BaseFixtureGenerator, QuickTester):
    fixture_sequence = [
        "object_class",
        "object_instance",
        "trainer_kwargs",
    ]

    def _all_objects(self) -> list[type]:
        return all_objects(
            object_types=self.object_type_filter,
            return_names=False,
            exclude_objects=self.exclude_objects,
        )

    @staticmethod
    def _check_required_dependencies(object_pkg):
        required_deps = object_pkg.get_class_tag("python_dependencies")
        if required_deps:
            try:
                if not _check_soft_dependencies(required_deps, severity="none"):
                    return False
            except Exception:
                return False
        return True

    @staticmethod
    def is_excluded(test_name: str, cls: type, instance_name: str = None) -> bool:
        skip_tests = cls.get_class_tag("tests:skip_tests", []) or []
        skip_params = cls.get_class_tag("tests:skip_params", []) or []

        if test_name in skip_tests:
            return True

        if instance_name is not None:
            if instance_name in skip_params:
                return True

            fixture_specific_test = f"{test_name}[{instance_name}]"
            if fixture_specific_test in skip_tests:
                return True

        return False

    def _generate_object_class(self, test_name: str, **kwargs) -> tuple[list, list]:
        """Filters the raw classes through the unified `is_excluded` check."""
        classes = []
        names = []
        for cls in self._all_objects():
            if not self.is_excluded(test_name, cls):
                classes.append(cls)
                names.append(cls.__name__)
        return classes, names

    def _generate_object_instance(self, test_name: str, **kwargs) -> tuple[list, list]:
        """Filters instantiated parameters through the unified `is_excluded` check."""
        classes, _ = self._generate_object_class(test_name)

        instances, names = [], []
        for cls in classes:
            param_list = cls.get_test_params()

            for i, params in enumerate(param_list):
                name = f"{cls.__name__}-{i}"

                if not self.is_excluded(test_name, cls, instance_name=name):
                    model_params = params.copy()
                    model_params.pop("trainer_kwargs", None)

                    inst = cls(**model_params)
                    inst._test_param_index = i

                    instances.append(inst)
                    names.append(name)

        return instances, names

    def _generate_trainer_kwargs(self, test_name: str, **kwargs) -> tuple[list, list]:
        """Generates the trainer_kwargs fixture perfectly paired to the instances."""

        if "object_instance" in kwargs:
            inst = kwargs["object_instance"]

            param_list = type(inst).get_test_params()

            i = getattr(inst, "_test_param_index", 0)
            t_kwargs = param_list[i].get("trainer_kwargs", {})

            return [t_kwargs], [""]

        classes, _ = self._generate_object_class(test_name)
        kwargs_list, names = [], []

        for cls in classes:
            param_list = cls.get_test_params()
            if isinstance(param_list, dict):
                param_list = [param_list]
            elif param_list is None:
                param_list = [{}]

            for i, params in enumerate(param_list):
                name = f"{cls.__name__}-{i}"
                if not self.is_excluded(test_name, cls, instance_name=name):
                    t_kwargs = params.get("trainer_kwargs", {})
                    kwargs_list.append(t_kwargs)
                    names.append(name)

        return kwargs_list, names
