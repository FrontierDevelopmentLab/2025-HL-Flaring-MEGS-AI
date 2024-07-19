#########################################################################################################
# MAIN
#########################################################################################################
configfile: "snakemake-config.yaml"

rule megsai_all:
    input:
        goes_csv = config["data"]["goes_path"]+"/"+config["data"]["goes_data"],
        eve_cdf = config["data"]["eve_path"]+"/"+config["data"]["eve_data"], 
        matches_csv = config["data"]["matches_path"]+"/"+config["data"]["matches_csv"], 
        # aia_stats = config["data"]["matches_path"]+"/"+config["data"]["matches_aia_path"], 
        eve_standardized = config["data"]["eve_path"]+"/"+config["data"]["eve_standardized"]
        

#########################################################################################################
# SETUP AND GENERATE TRAINING DATA
#########################################################################################################

## Download GOES soft X-ray flux data
rule download_goes_data:
    output: 
        goes_data = config["data"]["goes_path"]+"/"+config["data"]["goes_data"]
    params:
        output_path = config["data"]["goes_path"]
    shell:
        """
        mkdir -p {params.output_path} && 
        gsutil -m cp -r gs://us-4pieuvirradiance-dev-data/goes.csv {output.goes_data}
        """        

## Generate CDF file containing MEGS-A irradiance data
rule generate_eve_netcdf:
    input:
        eve_path = config["data"]["eve_path"]
    output:
        eve_data = config["data"]["eve_path"]+"/"+config["data"]["eve_data"]
    params:
        output_path = config["data"]["eve_path"]
    shell:
        """
        python irradiance/preprocess/generate_eve_netcdf.py \
        -start 2010-01-01T00:00:00 \
        -end 2011-01-01T00:00:00 \
        -path {input.eve_path} \
        -savepath {output.eve_data}
        """  

## Generates matches between MEGS-A data and AIA data
rule generate_matches_time:
    input:
        goes_data = config["data"]["goes_path"]+"/"+config["data"]["goes_data"],
        eve_data = config["data"]["eve_path"]+"/"+config["data"]["eve_data"],
        aia_path = config["data"]["aia_path"]
    output:
        matches_csv = config["data"]["matches_path"]+"/"+config["data"]["matches_csv"]
    params:
        eve_cutoff = config["data"]["eve_cutoff"],
        aia_cutoff = config["data"]["aia_cutoff"], 
        aia_wl = config["data"]["aia_wl"],
        matches_path = config["data"]["matches_path"],
        debug = config["debug"]
    shell:
        """
        mkdir -p {params.matches_path} &&
        python -m s4pi.irradiance.preprocess.generate_matches_time \
        -goes_path {input.goes_data} \
        -eve_path {input.eve_data} \
        -aia_path {input.aia_path} \
        -output_path {output.matches_csv} \
        -aia_wl {params.aia_wl} \
        -eve_cutoff {params.eve_cutoff} \
        -aia_cutoff {params.aia_cutoff} \
        -debug {params.debug}
        """

## Preprocess MEGS-A data
rule generate_eve_ml_ready:
    input:
        eve_data = config["data"]["eve_path"]+"/"+config["data"]["eve_data"],
        matches_csv = config["data"]["matches_path"]+"/"+config["data"]["matches_csv"]
    output:
        eve_standardized = config["data"]["eve_path"]+"/"+config["data"]["eve_standardized"],
        eve_stats = config["data"]["eve_path"]+"/"+config["data"]["eve_stats"]
        # eve_wl = config["data"]["megsa_path"]+"/"+config["data"]["megsa_wl"]
    shell:
        """
            python -m s4pi.irradiance.preprocess.generate_eve_ml_ready \
            -eve_path {input.eve_data} \
            -matches_table {input.matches_csv} \
            -output_data {output.eve_standardized} \
            -output_norm {output.eve_stats}
        """

## Generates donwnscaled stacks of the AIA channels
rule generate_euv_image_stacks:
    input:
        aia_path = config["data"]["aia_path"],
        matches_csv = config["data"]["matches_path"]+"/"+config["data"]["matches_csv"]
    params:
        aia_resolution = config["data"]["aia_resolution"],
        aia_reproject = config["data"]["aia_reproject"],
        matches_aia_path = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_aia_path"],
        debug = config["debug"]
    output:
        matches_csv = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_csv"],
        aia_stats = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["aia_stats"]
    shell:
        """
        mkdir -p {params.matches_aia_path} &&
        python -m s4pi.irradiance.preprocess.generate_euv_image_stacks \
        -aia_path {input.aia_path} \
        -aia_resolution {params.aia_resolution} \
        -aia_reproject {params.aia_reproject} \
        -aia_stats {output.aia_stats} \
        -matches_table {input.matches_csv} \
        -matches_output {output.matches_csv} \
        -matches_stacks {params.matches_aia_path} \
        -debug {params.debug}
        """
        

#########################################################################################################
# TRAIN & TEST MODEL
#########################################################################################################
