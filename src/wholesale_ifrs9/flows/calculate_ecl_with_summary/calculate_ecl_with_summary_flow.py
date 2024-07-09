if __name__ == "__main__":
    import tracdap.rt.launch as launch

    launch.launch_job("trac_poc/flows/calculate_ecl_with_summary/calculate_ecl_with_summary_flow_config.yaml", "trac_poc/config/sys_config.yaml", True)
