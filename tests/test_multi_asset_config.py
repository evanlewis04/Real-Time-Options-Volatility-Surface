from src.config.multi_asset_config import ConfigurationManager, ConfigurationPresets


def test_multi_asset_config_exports_imports_and_presets(tmp_path):
    path = tmp_path / "multi_asset_config.yaml"
    manager = ConfigurationManager(str(path))

    payload = manager.export_configuration("json")
    imported = ConfigurationManager(str(path))
    imported.import_configuration(payload, "json")

    summary = imported.get_configuration_summary()
    assert summary["assets"]["total_configured"] == len(imported.asset_configs)
    assert summary["system"]["cache_enabled"] is True

    conservative = ConfigurationPresets.conservative_trading()
    research = ConfigurationPresets.research_mode()
    assert conservative.portfolio_config.max_portfolio_var < research.portfolio_config.max_portfolio_var
    assert all(config.enabled for config in research.asset_configs.values())
