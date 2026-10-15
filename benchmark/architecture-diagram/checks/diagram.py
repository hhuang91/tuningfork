"""The check for the architecture-diagram tasks: the diagram the run wrote follows the skill's own rules.
One self-contained HTML file, its diagram an inline SVG, with no JavaScript and nothing external but Google
Fonts; every component the task names drawn in it; and the layout rule the skill marks CRITICAL — the
legend outside every boundary box, at least 20px below the lowest one — with the legend, like everything
else drawn, inside the SVG's visible area (its viewBox), where a browser shows it.

Coordinates are absolute: each element's transform and its groups' are applied. A boundary box is a dashed
rectangle that holds a component (a solid rectangle) inside it, as the skill draws regions and security
groups; the legend is the text that reads "Legend" and, when it sits in a group of its own, that group.
Prints what it found wrong, one finding a line; exit 0 on pass.

    diagram.py <task>
"""

from __future__ import annotations

import math
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

TASKS = {
    "pinboard": {
        "file": "pinboard-architecture.html",
        "components": {
            "the users": r"\busers?\b|browser|mobile",
            "CloudFront": r"cloud ?front",
            "the load balancer": r"load balanc|\b[ae]?lb\b",
            "the API service": r"\bapi\b",
            "PostgreSQL": r"postgres|\brds\b",
            "Redis": r"redis|elasticache",
            "the S3 bucket": r"\bs3\b",
            "Auth0": r"auth0",
        },
    },
}

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
UNDRAWN = {"defs", "marker", "pattern", "clippath", "mask", "symbol", "lineargradient", "radialgradient", "filter",
           "style", "title", "desc", "metadata"}
FONTS = ("fonts.googleapis.com", "fonts.gstatic.com")
EDGE = 1.0  # px a coordinate may stray outside the viewBox


class Node:
    def __init__(self, tag: str, attrs: dict[str, str], parent: Node | None):
        self.tag, self.attrs, self.parent = tag, attrs, parent
        self.children: list[Node] = []
        self.text: list[str] = []

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()

    def all_text(self) -> str:
        return "".join(self.text) + "".join(child.all_text() for child in self.children)


class Tree(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#document", {}, None)
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {name: value or "" for name, value in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.stack[-1].children.append(Node(tag, {name: value or "" for name, value in attrs}, self.stack[-1]))

    def handle_endtag(self, tag):
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth].tag == tag:
                del self.stack[depth:]
                return

    def handle_data(self, data):
        self.stack[-1].text.append(data)


NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def number(value: str | None) -> float | None:
    """A length as a plain number: "12", "12.5px"; a percentage or anything else is None."""
    if value is None:
        return None
    value = value.strip()
    match = re.fullmatch(r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(px)?", value)
    return float(match.group(1)) if match else None


def first_number(value: str | None) -> float | None:
    """The first of a list of lengths, as text's x and y may hold several."""
    if not value:
        return None
    return number(re.split(r"[\s,]+", value.strip())[0])


def multiply(m, n):
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = n
    return (a * a2 + c * b2, b * a2 + d * b2, a * c2 + c * d2, b * c2 + d * d2, a * e2 + c * f2 + e, b * e2 + d * f2 + f)


IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def transform_of(text: str | None):
    matrix = IDENTITY
    for name, args in re.findall(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)", text or ""):
        values = [float(v) for v in NUMBER.findall(args)]
        if name == "matrix" and len(values) == 6:
            local = tuple(values)
        elif name == "translate" and values:
            local = (1, 0, 0, 1, values[0], values[1] if len(values) > 1 else 0)
        elif name == "scale" and values:
            local = (values[0], 0, 0, values[1] if len(values) > 1 else values[0], 0, 0)
        elif name == "rotate" and values:
            angle = math.radians(values[0])
            cos, sin = math.cos(angle), math.sin(angle)
            local = (cos, sin, -sin, cos, 0, 0)
            if len(values) == 3:
                cx, cy = values[1], values[2]
                local = multiply(multiply((1, 0, 0, 1, cx, cy), local), (1, 0, 0, 1, -cx, -cy))
        elif name == "skewX" and values:
            local = (1, 0, math.tan(math.radians(values[0])), 1, 0, 0)
        elif name == "skewY" and values:
            local = (1, math.tan(math.radians(values[0])), 0, 1, 0, 0)
        else:
            continue
        matrix = multiply(matrix, local)
    return matrix


def apply(matrix, x, y):
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def styles(sheet: str) -> dict[str, dict[str, str]]:
    """Class rules of a stylesheet: `.name` (or `tag.name`) → its declarations."""
    rules: dict[str, dict[str, str]] = {}
    for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", re.sub(r"/\*.*?\*/", "", sheet, flags=re.S)):
        declarations = dict((k.strip().lower(), v.strip()) for k, _, v in
                            (part.partition(":") for part in body.split(";")) if k.strip())
        for selector in selectors.split(","):
            match = re.fullmatch(r"\s*[a-zA-Z]*\.([\w-]+)\s*", selector)
            if match:
                rules.setdefault(match.group(1), {}).update(declarations)
    return rules


def prop(node: Node, name: str, sheet: dict[str, dict[str, str]]) -> str | None:
    """A presentation property as the element has it: its style attribute, then its own attribute, then its
    classes' rules."""
    for part in node.attrs.get("style", "").split(";"):
        key, _, value = part.partition(":")
        if key.strip().lower() == name:
            return value.strip()
    if name in node.attrs:
        return node.attrs[name]
    for cls in node.attrs.get("class", "").split():
        if name in sheet.get(cls, {}):
            return sheet[cls][name]
    return None


def main() -> int:
    task = TASKS[sys.argv[1]]
    path = Path(task["file"])
    if not path.is_file():
        print(f"{task['file']} was not written in the project folder")
        return 1
    html = path.read_text(encoding="utf-8", errors="replace")
    problems: list[str] = []

    tree = Tree()
    tree.feed(html)
    tree.close()
    nodes = list(tree.root.walk())

    # One self-contained file: no JavaScript, nothing external but Google Fonts.
    if any(node.tag == "script" for node in nodes):
        problems.append("the file runs JavaScript (a <script> element); the skill says no JavaScript")
    elif any(name.startswith("on") for node in nodes for name in node.attrs):
        problems.append("the file runs JavaScript (an on… event attribute); the skill says no JavaScript")
    external = []
    for node in nodes:
        for name in ("src",) + (("href",) if node.tag == "link" else ()):
            value = node.attrs.get(name, "").strip()
            if re.match(r"(https?:)?//", value) and not any(host in value for host in FONTS):
                external.append(value)
    sheets = "".join(node.all_text() for node in nodes if node.tag == "style")
    sheets += "".join(node.attrs.get("style", "") for node in nodes)
    for url in re.findall(r"(?:@import\s+(?:url\()?|url\()\s*[\"']?((?:https?:)?//[^\"')\s]+)", sheets):
        if not any(host in url for host in FONTS):
            external.append(url)
    if external:
        problems.append(f"the file loads {external[0]}; the skill allows nothing external but Google Fonts")

    svgs = [node for node in nodes if node.tag == "svg"]
    if not svgs:
        problems.append("the file holds no inline SVG diagram")
        print("\n".join(problems))
        return 1
    svg = max(svgs, key=lambda node: sum(1 for _ in node.walk()))
    box = [number(v) for v in re.split(r"[\s,]+", svg.attrs.get("viewbox", "").strip())] if svg.attrs.get("viewbox") else []
    if len(box) == 4 and None not in box and box[2] > 0 and box[3] > 0:
        left, top, width, height = box
    elif number(svg.attrs.get("width")) and number(svg.attrs.get("height")):
        left, top, width, height = 0.0, 0.0, number(svg.attrs.get("width")), number(svg.attrs.get("height"))
    else:
        problems.append("the diagram's SVG has neither a viewBox nor a width and height")
        print("\n".join(problems))
        return 1
    right, bottom = left + width, top + height
    sheet = styles(sheets)

    # Every drawn element in absolute coordinates.
    rects, texts, points = [], [], []

    def visit(node: Node, matrix, hidden: bool):
        local = multiply(matrix, transform_of(node.attrs.get("transform"))) if node is not svg else matrix
        hidden = hidden or node.tag in UNDRAWN
        if not hidden:
            if node.tag == "rect":
                x, y = number(node.attrs.get("x", "0")), number(node.attrs.get("y", "0"))
                w, h = number(node.attrs.get("width")), number(node.attrs.get("height"))
                if None not in (x, y, w, h):
                    corners = [apply(local, cx, cy) for cx, cy in ((x, y), (x + w, y), (x, y + h), (x + w, y + h))]
                    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
                    dash = (prop(node, "stroke-dasharray", sheet) or "").strip().lower()
                    rects.append({"node": node, "x0": min(xs), "y0": min(ys), "x1": max(xs), "y1": max(ys),
                                  "dashed": dash not in ("", "none", "0")})
                    points.extend([(min(xs), min(ys), "a rectangle"), (max(xs), max(ys), "a rectangle")])
            elif node.tag == "text":
                x, y = first_number(node.attrs.get("x")), first_number(node.attrs.get("y"))
                if y is None:
                    span = next((n for n in node.walk() if n.tag == "tspan" and n.attrs.get("y")), None)
                    if span is not None:
                        x, y = first_number(span.attrs.get("x")) or x, first_number(span.attrs.get("y"))
                ax, ay = apply(local, x or 0.0, y or 0.0)
                words = " ".join(node.all_text().split())
                texts.append({"node": node, "x": ax, "y": ay, "text": words})
                points.append((ax, ay, f"the text {words[:40]!r}"))
                return  # its tspans are part of it
            elif node.tag == "line":
                for a, b in (("x1", "y1"), ("x2", "y2")):
                    x, y = number(node.attrs.get(a, "0")), number(node.attrs.get(b, "0"))
                    if None not in (x, y):
                        points.append((*apply(local, x, y), "a line"))
            elif node.tag == "circle":
                cx, cy, r = (number(node.attrs.get(k, "0")) for k in ("cx", "cy", "r"))
                if None not in (cx, cy, r):
                    for dx, dy in ((-r, -r), (r, r)):
                        points.append((*apply(local, cx + dx, cy + dy), "a circle"))
        for child in node.children:
            visit(child, local, hidden)

    visit(svg, IDENTITY, False)

    # Every component the task names, drawn.
    words = "\n".join(t["text"] for t in texts).lower()
    for name, pattern in task["components"].items():
        if not re.search(pattern, words):
            problems.append(f"{name} is not in the diagram")

    # The boundary boxes: dashed rectangles holding a component — a solid box the size of one (the skill's
    # services are 60px high; a legend's swatches are a few pixels).
    def inside(inner, outer):
        return (inner["x0"] >= outer["x0"] - 0.5 and inner["y0"] >= outer["y0"] - 0.5
                and inner["x1"] <= outer["x1"] + 0.5 and inner["y1"] <= outer["y1"] + 0.5)

    solid = [r for r in rects if not r["dashed"] and r["x1"] - r["x0"] >= 40 and r["y1"] - r["y0"] >= 25]
    boundaries = [r for r in rects if r["dashed"] and any(inside(c, r) for c in solid if c is not r)]
    if not boundaries:
        problems.append("no boundary box: the region and security groups the task names are not drawn as the "
                        "skill draws boundaries (dashed rectangles around their components)")

    # The legend: outside every boundary box, at least 20px below the lowest one. In the SVG, its title; or
    # set in the page below the diagram, which no boundary box can reach.
    title = next((t for t in texts if re.fullmatch(r"legend\b.*|key:?", t["text"], re.I)), None)
    order = {id(node): index for index, node in enumerate(nodes)}
    in_svg = {id(node) for node in svg.walk()}
    below_page = any(order[id(node)] > order[id(svg)] and id(node) not in in_svg
                     and re.fullmatch(r"\s*(legend\b.*|key:?)\s*", "".join(node.text), re.I | re.S)
                     for node in nodes if node.tag not in ("style", "script", "title"))
    if title is None and not below_page:
        problems.append("the diagram has no legend")
    elif title is not None and boundaries:
        lowest = max(r["y1"] for r in boundaries)
        if title["y"] < lowest + 20 - 0.01:
            problems.append(f"the legend is at y={title['y']:g}, not 20px below the lowest boundary box, whose "
                            f"bottom is at y={lowest:g}")
        # When the legend has a group of its own, all of it.
        group = title["node"].parent
        while group is not None and group is not svg and group.tag != "g":
            group = group.parent
        if group is not None and group is not svg:
            members = {id(node) for node in group.walk()}
            if not any(id(r["node"]) in members for r in boundaries):
                above = [r for r in rects if id(r["node"]) in members and r["y0"] < lowest - 0.5]
                above += [t for t in texts if id(t["node"]) in members and t["y"] < lowest - 0.5]
                if above:
                    problems.append(f"part of the legend sits above the lowest boundary box's bottom "
                                    f"(y={lowest:g})")

    # Everything drawn inside the visible area.
    outside = [(x, y, what) for x, y, what in points
               if x < left - EDGE or x > right + EDGE or y < top - EDGE or y > bottom + EDGE]
    if outside:
        x, y, what = outside[0]
        problems.append(f"{len(outside)} point(s) of the diagram lie outside its visible area (viewBox "
                        f"{left:g} {top:g} {width:g} {height:g}), where a browser cuts them off; the first: "
                        f"{what} at ({x:g}, {y:g})")

    if problems:
        print("\n".join(problems))
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
