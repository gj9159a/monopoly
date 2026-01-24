from pathlib import Path
import yaml

def _load_card_ids(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise AssertionError(f"{path.name} должен быть списком карточек")
    return {str(item['id']) for item in data if isinstance(item, dict) and item.get('id')}


def test_cards_texts_template_covers_all_ids():
    data_dir = Path(__file__).resolve().parents[1] / "monopoly" / "data"
    chance_ids = _load_card_ids(data_dir / "cards_chance.yaml")
    community_ids = _load_card_ids(data_dir / "cards_community.yaml")
    all_ids = chance_ids | community_ids

    template_path = data_dir / "cards_texts_ru_official.yaml.template"
    template = yaml.safe_load(template_path.read_text(encoding='utf-8'))
    assert isinstance(template, dict)
    template_ids = {str(key) for key in template.keys() if key is not None}

    assert all_ids == template_ids
