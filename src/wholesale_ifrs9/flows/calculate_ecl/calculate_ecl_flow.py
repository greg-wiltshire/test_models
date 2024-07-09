if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_job("trac_poc/flows/calculate_ecl/calculate_ecl_flow_config.yaml", "trac_poc/config/sys_config.yaml", True)
