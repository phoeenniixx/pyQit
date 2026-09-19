"""``.. object-overview::``, a filterable table of ``object_overview()``."""

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

from pyqit.base import object_overview


class ObjectOverview(Directive):
    def run(self):
        lines = [
            ".. raw:: html",
            "",
            '   <input id="object-filter" type="search" '
            'placeholder="Filter, e.g. model hybrid or estimator_type: regressor" '
            'style="width: 100%; margin-bottom: 1em; padding: 0.4em">',
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "   :class: object-overview",
            "",
            "   * - Object",
            "     - Type",
            "     - Tags",
        ]
        for row in object_overview():
            tags = ", ".join(f"``{k}: {v}``" for k, v in row["tags"].items())
            lines += [
                f"   * - :class:`~{row['path']}`",
                f"     - {row['object_type']}",
                f"     - {tags}",
            ]
        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(StringList(lines), self.content_offset, node)
        return node.children


def setup(app):
    app.add_directive("object-overview", ObjectOverview)
    app.add_js_file("overview.js")
    return {"parallel_read_safe": True}
