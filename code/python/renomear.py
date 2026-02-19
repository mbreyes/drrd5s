# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:01:29 2025

@author: lilio
"""
import os
import sys

def renomear_arquivos(base_dir):
    """
    Percorre todas as subpastas do experimento AY,
    localiza os arquivos com extensão .999 e os renomeia para .0XX,
    onde XX representa o número da sessão baseado na ordem da data.
    """
    # Percorre todas as pastas AY1 a AY5
    for az_folder in ["AZ0","AZ1", "AZ2", "AZ3", "AZ4"]:
        az_path = os.path.join(base_dir, az_folder)

        # Verifica se a pasta existe
        if not os.path.exists(az_path):
            print(f"Pasta não encontrada: {az_path}")
            continue

        # Lista e ordena as pastas por data
        data_folders = sorted([d for d in os.listdir(az_path) if os.path.isdir(os.path.join(az_path, d))])

        # Percorre cada pasta de data e renomeia os arquivos
        for session_number, data_folder in enumerate(data_folders, start=1):
            data_path = os.path.join(az_path, data_folder, "Data")

            if not os.path.exists(data_path):
                print(f"Pasta 'Data' não encontrada em: {data_path}")
                continue

            # Define o novo sufixo de acordo com o número da sessão
            new_suffix = f".{session_number:03d}"

            # Renomeia todos os arquivos que terminam com .999
            for file in os.listdir(data_path):
                if file.endswith(".999"):
                    old_file = os.path.join(data_path, file)
                    new_file = os.path.join(data_path, file.replace(".999", new_suffix))

                    # Renomeia o arquivo
                    os.rename(old_file, new_file)
                    print(f"Renomeado: {old_file} -> {new_file}")

    print("Processo concluído!")

if __name__ == "__main__":
    #if len(sys.argv) != 2:
     #   print("Uso: python renomear_AZ.py </Users/ailacamara/Downloads/DRRD/AZ>")
      #  sys.exit(1)

    caminho_base = "/Users/ailacamara/Downloads/DRRD/data/raw/AZ"
    renomear_arquivos(caminho_base)
           
