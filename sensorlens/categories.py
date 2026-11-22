"""类别体系：把各家标注里五花八门的类别名归到统一的大类，用于配色、筛选和统计。

规则来自 adapters/categories.json（可按公司标签体系修改），未配置时使用内置默认值。
匹配顺序：精确别名 → 关键词包含；都不匹配归为 other。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_GROUPS = [
    {"name": "vehicle_4w", "title": "四轮车", "color": "#40a9ff",
     "aliases": ["car", "suv", "van", "truck", "bus", "trailer", "construction_vehicle", "pickup", "minibus",
                 "vehicle", "vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.trailer",
                 "vehicle.construction", "vehicle.emergency", "汽车", "小汽车", "轿车", "卡车", "货车", "客车",
                 "公交车", "面包车", "工程车", "挂车"],
     "keywords": ["car", "truck", "bus", "van", "suv", "trailer", "汽车", "卡车", "货车", "客车", "轿车"]},
    {"name": "vehicle_2w", "title": "两轮车", "color": "#ffa940",
     "aliases": ["bicycle", "motorcycle", "cyclist", "motorcyclist", "scooter", "tricycle", "moped",
                 "vehicle.bicycle", "vehicle.motorcycle", "自行车", "摩托车", "电动车", "电瓶车", "三轮车",
                 "骑行者", "两轮车"],
     "keywords": ["bicycle", "bike", "motor", "cycl", "scooter", "tricycle", "自行车", "摩托", "电动车",
                  "三轮", "骑"]},
    {"name": "pedestrian", "title": "行人", "color": "#ff4d4f",
     "aliases": ["pedestrian", "person", "people", "child", "human", "walker",
                 "human.pedestrian.adult", "human.pedestrian.child", "行人", "人", "儿童"],
     "keywords": ["pedestrian", "person", "human", "people", "child", "行人", "儿童"]},
    {"name": "obstacle", "title": "障碍物", "color": "#73d13d",
     "aliases": ["cone", "traffic_cone", "barrier", "bollard", "barrel", "debris", "obstacle", "animal",
                 "movable_object.trafficcone", "movable_object.barrier", "movable_object.debris",
                 "static_object.bicycle_rack", "锥桶", "雪糕筒", "水马", "护栏", "障碍物", "石墩", "动物"],
     "keywords": ["cone", "barrier", "bollard", "barrel", "debris", "obstacle", "锥", "障碍", "水马",
                  "护栏", "石墩"]},
    {"name": "road", "title": "车道线/路面", "color": "#e8e8e8",
     "aliases": ["lane", "lane_solid", "lane_dashed", "curb", "stop_line", "crosswalk", "road_edge",
                 "车道线", "实线", "虚线", "路沿", "停止线", "斑马线"],
     "keywords": ["lane", "curb", "line", "crosswalk", "road", "车道", "线", "路沿", "斑马"]},
]
OTHER = {"name": "other", "title": "其他", "color": "#b37feb"}


class Categories:
    def __init__(self, groups: list[dict] | None = None):
        self.groups = [dict(g) for g in (groups or DEFAULT_GROUPS)]
        self._alias: dict[str, str] = {}
        for g in self.groups:
            for a in g.get("aliases", []):
                self._alias.setdefault(self._norm(a), g["name"])
        self._by_name = {g["name"]: g for g in self.groups}
        self._by_name[OTHER["name"]] = OTHER
        self._cache: dict[str, str] = {}

    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[\s\-]+", "_", str(s).strip().lower())

    def group_of(self, label: str) -> str:
        key = self._norm(label)
        if key in self._cache:
            return self._cache[key]
        g = self._alias.get(key)
        if g is None:
            g = next((gr["name"] for gr in self.groups
                      if any(self._norm(k) in key for k in gr.get("keywords", []))), OTHER["name"])
        self._cache[key] = g
        return g

    def color(self, group: str) -> str:
        return self._by_name.get(group, OTHER)["color"]

    def describe(self) -> list[dict]:
        return [{"name": g["name"], "title": g.get("title", g["name"]), "color": g["color"]}
                for g in self.groups + [OTHER]]


_DEFAULT = Categories()


def load_categories(path: str | Path | None) -> Categories:
    if path and Path(path).is_file():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return Categories(data.get("groups"))
    return _DEFAULT
