from typing import List
import requests
import re
import os


def get_list_url(url: str) -> List[str]:
    response = requests.get(url)
    response.raise_for_status()
    # Use a regex pattern to find all href links in the page
    files = re.findall(r'href="([^"]+).fna.gz"', response.text)

    return [os.path.join(url, f"{file}.fna.gz") for file in files]


if __name__ == "__main__":
    fasta_files = get_list_url("https://ftp.ncbi.nlm.nih.gov/refseq/release/complete/")
    print("\n".join(fasta_files))
    print(f"Total FASTA files found: {len(fasta_files)}")
    # Save the list to a file
    # with open('fasta_files.txt', 'w') as f:
    #     for file_path in fasta_files:
    #         f.write(f'{file_path}\n')
