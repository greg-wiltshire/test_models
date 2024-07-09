if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_job("trac_poc/flows/generate_scenarios/generate_scenarios_flow_config.yaml", "trac_poc/config/sys_config.yaml", True)
