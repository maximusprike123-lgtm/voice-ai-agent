from datetime import time
from pathlib import Path

import pytest

from agent.business import BusinessConfigError, Weekday, load_business_config

REPO_CONFIG = Path(__file__).parent.parent / "config" / "business.yaml"

VALID_YAML = """
name: "Тест"
address: "Москва"
greeting: "Здравствуйте!"
hours:
  mon: { open: "10:00", close: "21:00" }
  tue: { open: "10:00", close: "21:00" }
  wed: { open: "10:00", close: "21:00" }
  thu: { open: "10:00", close: "21:00" }
  fri: { open: "10:00", close: "21:00" }
  sat: { open: "10:00", close: "20:00" }
  sun: null
services:
  - { id: wash, name: "Мойка", price_from: 3000, price_to: 5000 }
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "business.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_repo_config_is_valid():
    config = load_business_config(REPO_CONFIG)
    assert config.services
    assert config.hours[Weekday.SUN] is None


def test_valid_config(tmp_path):
    config = load_business_config(write(tmp_path, VALID_YAML))
    assert config.hours[Weekday.MON].open == time(10, 0)
    assert config.service_by_id("wash").name == "Мойка"
    assert config.service_by_id("nope") is None


def test_unquoted_time_is_rejected(tmp_path):
    # YAML 1.1 reads unquoted 10:00 as the integer 600.
    text = VALID_YAML.replace('mon: { open: "10:00"', "mon: { open: 10:00")
    with pytest.raises(BusinessConfigError, match="quoted"):
        load_business_config(write(tmp_path, text))


def test_missing_weekday_is_rejected(tmp_path):
    text = VALID_YAML.replace("  sun: null\n", "")
    with pytest.raises(BusinessConfigError, match="sun"):
        load_business_config(write(tmp_path, text))


def test_open_after_close_is_rejected(tmp_path):
    text = VALID_YAML.replace('mon: { open: "10:00"', 'mon: { open: "22:00"')
    with pytest.raises(BusinessConfigError, match="earlier"):
        load_business_config(write(tmp_path, text))


def test_inverted_price_range_is_rejected(tmp_path):
    text = VALID_YAML.replace("price_to: 5000", "price_to: 1000")
    with pytest.raises(BusinessConfigError, match="price_to"):
        load_business_config(write(tmp_path, text))


def test_duplicate_service_ids_are_rejected(tmp_path):
    text = VALID_YAML + '  - { id: wash, name: "Ещё мойка", price_from: 1000 }\n'
    with pytest.raises(BusinessConfigError, match="duplicate"):
        load_business_config(write(tmp_path, text))


def test_reserved_service_id_other_is_rejected(tmp_path):
    text = VALID_YAML + '  - { id: other, name: "Прочее", price_from: 1000 }\n'
    with pytest.raises(BusinessConfigError, match="'other' is reserved"):
        load_business_config(write(tmp_path, text))


def test_unknown_field_is_rejected(tmp_path):
    with pytest.raises(BusinessConfigError, match="phone_number"):
        load_business_config(write(tmp_path, VALID_YAML + 'phone_number: "+7"\n'))


def test_missing_file(tmp_path):
    with pytest.raises(BusinessConfigError, match="cannot read"):
        load_business_config(tmp_path / "absent.yaml")


def test_broken_yaml(tmp_path):
    with pytest.raises(BusinessConfigError, match="invalid YAML"):
        load_business_config(write(tmp_path, "name: [unclosed"))
