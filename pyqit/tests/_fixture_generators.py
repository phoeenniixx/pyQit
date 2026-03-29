import inspect
import traceback

import pytest
from skbase.testing import BaseFixtureGenerator as _BaseFixtureGenerator, QuickTester
from skbase.utils.dependencies import _check_soft_dependencies

from pyqit.base.base_object import all_objects


class BaseFixtureGenerator(_BaseFixtureGenerator, QuickTester):
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

    def _generate_object_class(self, test_name: str) -> tuple[list, list]:
        """Filters the raw classes through the unified `is_excluded` check."""
        classes = []
        names = []
        for cls in self._all_objects():
            if not self.is_excluded(test_name, cls):
                classes.append(cls)
                names.append(cls.__name__)
        return classes, names

    def _generate_object_instance(self, test_name: str) -> tuple[list, list]:
        """Filters instantiated parameters through the unified `is_excluded` check."""
        classes, _ = self._generate_object_class(test_name)

        instances, names = [], []
        for cls in classes:
            cls_insts, cls_names = cls.create_test_instances_and_names()

            for inst, name in zip(cls_insts, cls_names):
                if not self.is_excluded(test_name, cls, instance_name=name):
                    instances.append(inst)
                    names.append(name)

        return instances, names
