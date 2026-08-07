import os
import time

def run_abaqus_job(input_file: str):
    """
    Runs an Abaqus job using the provided input file.
    """
    job_name = os.path.splitext(os.path.basename(input_file))[0]
    abaqus_command = f"abaqus interactive double analysis ask_delete=off job={job_name} user=uel_nlmech_pk1.for output_precision=full interactive"

    # Run the command using os.system
    print(f"Running Abaqus job: {abaqus_command}")
    os.system(abaqus_command)

if __name__ == "__main__":
    start = time.perf_counter()

    run_abaqus_job("UEL_600_elements_modified.inp")

    #UEL_600_elements_modified
    #UEL_600_elements_modified_sem_tg:18.845398 seconds
    #Meu_input_file-19.629564
    #Meu_input_file_sem_tg-18.716397
    end = time.perf_counter()
    print(f"Time taken: {end - start:.6f} seconds")