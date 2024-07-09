if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_job("wholesale_ifrs9/flows/generate_scenarios_and_pd_curves/generate_scenarios_and_pd_curves_flow_config.yaml", "wholesale_ifrs9/config/sys_config.yaml", True)
