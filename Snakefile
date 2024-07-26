#########################################################################################################
# MAIN
#########################################################################################################
configfile: "snakemake-config.yaml"
# Draw: snakemake --forceall --dag | dot -Tpng > dag.png

# TODO: Maybe remove "data" section from config (config['data']) file to make it simpler to read
# General rule to generate all data
rule megsai_all:
    input:
        # checkpoints = expand(config["model"]["checkpoint_path"]+"/{instrument}/"+config["model"]["checkpoint_file"]+".ckpt", instrument=config["model"]["instruments"]),
        checkpoint = expand(config["model"]["checkpoint_path"]+"/{instrument}/"+f"{config['data']['eve_type']}_{config['data']['eve_instrument']}/"+config["model"]["checkpoint_file"], instrument=config["model"]["instruments"]),
        maven_lvl3_data = f"{config['data']['maven_lvl3_dir']}/{config['data']['maven_lvl3_data']}",
        fismp_earth_data = f"{config['data']['fismp_dir']}/{config['data']['fismp_earth_data']}",
        fismp_mars_data = f"{config['data']['fismp_dir']}/{config['data']['fismp_mars_data']}",
        fism2_data = expand(f"{config['data']['fism2_dir']}/"+"{fism2_type}"+f"/FISM_2014001_{config['data']['fism2_version']}.sav", fism2_type=config['data']['fism2_type']),
        eve_standardized = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_eve_subdir']}/"+
                           f"{config['data']['eve_type']}_"+
                           f"{config['data']['eve_instrument']}_{config['data']['eve_standardized']}",
        updated_matches_csv= f"{config['data']['matches_dir']}/{config['data']['preprocess_aia_subdir']}_"+
                             f"{config['data']['aia_resolution']}_stacks_{config['data']['eve_type']}_"+
                             f"{config['data']['eve_instrument']}_{config['data']['matches_csv']}"
        

# Draw the DAG of the pipeline
rule draw_dag:
    output:
        png='dag.png'
    shell:
        "snakemake --forceall --dag | dot -Tpng > {output.png}"

#########################################################################################################
# SETUP AND GENERATE TRAINING DATA
#########################################################################################################

## Download GOES soft X-ray flux data
rule download_goes_data:
    output: 
        goes_data = f"{config['data']['goes_dir']}/{config['data']['goes_data']}"
    params:
        goes_dir = config['data']['goes_dir']
    shell:
        """
        mkdir -p {params.goes_dir} && 
        gsutil -m cp -r gs://us-spi3s-landing/megs_ai/observational_data/GOES/goes.csv {output.goes_data}
        """        

## Download EVE irradiance data
rule download_eve_data:
    output:
        eve_file = f"{config['data']['eve_dir']}/{config['data']['eve_type']}_L{config['data']['eve_level']}"+
                   "_2014001_008_01.fit.gz"
    params:
        eve_type = config['data']['eve_type'],
        eve_level = config['data']['eve_level'],
        eve_dir = config['data']['eve_dir']
    shell:
        """
        mkdir -p {params.eve_dir} &&
        python -m irradiance.data.download_eve_data \
        -start 2010-01-01T00:00:00 \
        -end 2015-01-01T00:00:00 \
        -type {params.eve_type} \
        -level {params.eve_level} \
        -save_dir {params.eve_dir} 
        """

rule download_maven_data:
    output:
        maven_lvl3_data = f"{config['data']['maven_lvl3_dir']}/{config['data']['maven_lvl3_data']}"
    params:
        maven_dir = config['data']['maven_dir'],
        maven_lvl2_dir = config['data']['maven_lvl2_dir'],
        maven_lvl3_dir = config['data']['maven_lvl3_dir']
    shell:
        """
        mkdir -p {params.maven_dir} && 
        mkdir -p {params.maven_lvl2_dir} &&
        mkdir -p {params.maven_lvl3_dir} &&
        gsutil -m cp -r gs://us-spi3s-landing/megs_ai/observational_data/MAVEN/level3/mvn_euv_l3_daily.csv {output.maven_lvl3_data}
        """

rule download_fismp_data:
    output:
        fismp_earth_data = f"{config['data']['fismp_dir']}/{config['data']['fismp_earth_data']}",
        fismp_mars_data = f"{config['data']['fismp_dir']}/{config['data']['fismp_mars_data']}"
    params:
        fismp_dir = config['data']['fismp_dir']
    shell:
        """
        mkdir -p {params.fismp_dir} && 
        gsutil -m cp -r gs://us-spi3s-landing/megs_ai/observational_data/FISM-P/fism_p_spectrum_earth_l2v01_r00_l3v01_r00_prelim.nc {output.fismp_earth_data} && 
        gsutil -m cp -r gs://us-spi3s-landing/megs_ai/observational_data/FISM-P/fism_p_spectrum_mars_l2v01_r00_l3v01_r00_prelim.nc {output.fismp_mars_data}
        """

rule download_fism2_data:
    output:
        fism2_data = f"{config['data']['fism2_dir']}/"+"{fism2_type}"+f"/FISM_2014001_{config['data']['fism2_version']}.sav"
    params:
        fism2_dir = config['data']['fism2_dir'],
        fism2_type = "{fism2_type}",
        fism2_version = config['data']['fism2_version'],
        fism2_url = config['data']['fism2_url']
    shell:
        """
        mkdir -p {params.fism2_dir} &&
        python -m irradiance.data.download_fism2 \
        -start 2010-01-01T00:00:00 \
        -end   2014-05-10T23:59:59 \
        -type {params.fism2_type} \
        -version {params.fism2_version} \
        -url {params.fism2_url} \
        -save_dir {params.fism2_dir}
        """

## Generate CDF file containing EVE irradiance data
rule generate_eve_netcdf:
    input:
        eve_dir = config['data']['eve_dir'],
        eve_file = f"{config['data']['eve_dir']}/{config['data']['eve_type']}_L{config['data']['eve_level']}"+
                   "_2014001_008_01.fit.gz"
    output:
        eve_data = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_eve_subdir']}/{config['data']['eve_type']}_{config['data']['eve_instrument']}_"+
                   f"{config['data']['eve_data']}"
    params:
        eve_type = config['data']['eve_type'],
        eve_level = config['data']['eve_level'],
        eve_instrument = config['data']['eve_instrument'],
    shell:
        """
        python -m irradiance.preprocess.generate_eve_netcdf \
        -start 2010-01-01T00:00:00 \
        -end 2015-01-01T00:00:00 \
        -type {params.eve_type} \
        -level {params.eve_level} \
        -instrument {params.eve_instrument} \
        -data_dir {input.eve_dir} \
        -save_path {output.eve_data}
        """  

## Generates matches in time between EVE data and AIA data
rule generate_matches_time:
    input:
        goes_data = config['data']['goes_dir']+"/"+config['data']['goes_data'],
        eve_data = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_eve_subdir']}/{config['data']['eve_type']}_{config['data']['eve_instrument']}_"+
                   f"{config['data']['eve_data']}",
        imager_dir = config['data']['aia_dir']
    output:
        matches_csv = f"{config['data']['matches_dir']}/{config['data']['eve_type']}_"+
                      f"{config['data']['eve_instrument']}_{config['data']['matches_csv']}"
    params:
        eve_to_imager_dt = config['data']['eve_cutoff'],
        imager_dt = config['data']['aia_cutoff'],
        imager_wl = config['data']['aia_wl'],
        matches_dir = config['data']['matches_dir']
    shell:
        """
        mkdir -p {params.matches_dir} &&
        python -m irradiance.preprocess.generate_matches_time \
        -imager_dir {input.imager_dir} \
        -imager_wl {params.imager_wl} \
        -imager_dt {params.imager_dt} \
        -goes_data {input.goes_data} \
        -eve_data {input.eve_data} \
        -output_path {output.matches_csv} \
        -eve_to_imager_dt {params.eve_to_imager_dt}
        """

## Standardizes EVE data (for which matches were found) and generates statistics
rule generate_eve_standardized:
    input:
        eve_data = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_eve_subdir']}/{config['data']['eve_type']}_{config['data']['eve_instrument']}_"+
                   f"{config['data']['eve_data']}",
        matches_csv = f"{config['data']['matches_dir']}/{config['data']['eve_type']}_"+
                      f"{config['data']['eve_instrument']}_{config['data']['matches_csv']}"
    output:
        eve_standardized = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_eve_subdir']}/"+
                           f"{config['data']['eve_type']}_"+
                           f"{config['data']['eve_instrument']}_{config['data']['eve_standardized']}",
        eve_stats = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_eve_subdir']}/"+
                    f"{config['data']['eve_type']}_"+
                    f"{config['data']['eve_instrument']}_{config['data']['eve_stats']}"
    params:
        matches_eve_dir = f"{config['data']['matches_dir']}"
    shell:
        """
            mkdir -p {params.matches_eve_dir} &&
            python -m irradiance.preprocess.generate_eve_standardized \
            -eve_data {input.eve_data} \
            -matches_csv {input.matches_csv} \
            -output_data {output.eve_standardized} \
            -output_stats {output.eve_stats}
        """

## Generates donwnscaled stacks of the AIA channels
rule generate_imager_stacks:
    input:
        imager_dir = config['data']['aia_dir'],
        matches_csv = f"{config['data']['matches_dir']}/{config['data']['eve_type']}_"+
                      f"{config['data']['eve_instrument']}_{config['data']['matches_csv']}"
    params:
        suffix = f"{config['data']['eve_type']}_{config['data']['eve_instrument']}",
        imager_resolution = config['data']['aia_resolution'],
        imager_reproject = config['data']['aia_reproject'],
        matches_imager_dir = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_aia_subdir']}_{config['data']['aia_resolution']}"
    output:
        matches_csv = f"{config['data']['matches_dir']}/{config['data']['preprocess_aia_subdir']}_"+
                      f"{config['data']['aia_resolution']}_stacks_{config['data']['eve_type']}_"+
                      f"{config['data']['eve_instrument']}_{config['data']['matches_csv']}",
        imager_stats = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_aia_subdir']}_"+
                       f"{config['data']['aia_resolution']}_{config['data']['eve_type']}_"+
                       f"{config['data']['eve_instrument']}_{config['data']['aia_stats']}"
    shell:
        """
        mkdir -p {params.matches_imager_dir} &&
        python -m irradiance.preprocess.generate_imager_stacks \
        -imager_path {input.imager_dir} \
        -imager_resolution {params.imager_resolution} \
        -imager_reproject {params.imager_reproject} \
        -imager_stats {output.imager_stats} \
        -matches_csv {input.matches_csv} \
        -matches_output {output.matches_csv} \
        -matches_stacks {params.matches_imager_dir} \
        -norm_suffix {params.suffix}
        """
        

#########################################################################################################
# TRAIN & TEST MODEL
#########################################################################################################

rule megsai_train:
    input:
        matches_table = f"{config['data']['matches_dir']}/{config['data']['preprocess_aia_subdir']}_"+
                        f"{config['data']['aia_resolution']}_stacks_{config['data']['eve_type']}_"+
                        f"{config['data']['eve_instrument']}_{config['data']['matches_csv']}",
        eve_converted_data = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_eve_subdir']}/"+
                             f"{config['data']['eve_type']}_"+
                             f"{config['data']['eve_instrument']}_{config['data']['eve_standardized']}",
        eve_stats = f"{config['data']['preprocess_dir']}/{config['data']['preprocess_eve_subdir']}/"+
                    f"{config['data']['eve_type']}_"+
                    f"{config['data']['eve_instrument']}_{config['data']['eve_stats']}"
    params:
        instrument = "{instrument}",
        config_file = config["model"]["config_file"],
        checkpoint_path = config["model"]["checkpoint_path"]+"/{instrument}/"+f"{config['data']['eve_type']}_{config['data']['eve_instrument']}"
        #  checkpoint_file = config["model"]["checkpoint_path"]+"/{instrument}/"+f"{config['data']['eve_type']}_{config['data']['eve_instrument']}/"+config["model"]["checkpoint_file"]
    output: 
        checkpoint = config["model"]["checkpoint_path"]+"/{instrument}/"+f"{config['data']['eve_type']}_{config['data']['eve_instrument']}/"+config["model"]["checkpoint_file"]
    resources:
        nvidia_gpu = 1
    shell:
        """
        mkdir -p {params.checkpoint_path} &&
        python -m irradiance.train \
        -checkpoint {output.checkpoint} \
        -model {params.config_file} \
        -matches_table {input.matches_table} \
        -eve_data {input.eve_converted_data} \
        -eve_norm {input.eve_stats} \
        -instrument {params.instrument}
        """
