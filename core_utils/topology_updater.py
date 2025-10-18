import re
import os
from config_params.config import Config

def create_system_topology(output_dir, top_path, itp_files):
    top_content = ""
    default_itps = Config.get_param('additional_itp_files')
    all_itps = list(dict.fromkeys(default_itps + itp_files))
    
    for itp_file in all_itps:
        # Always use absolute paths for include files as requested.
        abs_path = os.path.abspath(itp_file)
        top_content += f'#include "{abs_path}"\n'

    top_content += "\n[ system ]\nMy System\n\n[ molecules ]\n; Compound        #mols\n"
    with open(top_path, 'w') as f:
        f.write(top_content)
    print(f"성공적으로 토폴로지 파일 '{top_path}'을 생성했습니다.")

def update_topology_molecules(topology_file, molecule_counts, additional_itp_includes=None):
    """
    Updates the [ molecules ] section of a GROMACS topology file.

    Args:
        topology_file (str): Path to the .top file.
        molecule_counts (dict): A dictionary where keys are molecule names and
                                values are their counts.
        additional_itp_includes (list, optional): A list of paths to .itp files to include.
    """
    if not os.path.exists(topology_file):
        print(f"경고: 토폴로지 파일 '{topology_file}'을(를) 찾을 수 없어 업데이트를 건너뜁니다.")
        return

    with open(topology_file, 'r') as f:
        lines = f.readlines()

    # --- Include additional ITP files ---
    if additional_itp_includes:
        include_section_end_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                include_section_end_index = i
            elif line.strip().startswith('[ system ]'):
                break
        
        new_includes = []
        for itp_path in additional_itp_includes:
            include_line = f'#include "{itp_path}"\n'
            # Avoid adding duplicate includes
            if not any(include_line.strip() == l.strip() for l in lines):
                new_includes.append(include_line)
        
        if new_includes:
            insert_index = include_section_end_index + 1 if include_section_end_index != -1 else 0
            lines[insert_index:insert_index] = new_includes
            print(f"'{topology_file}'에 추가 ITP 파일들을 포함했습니다.")

    # --- Update [ molecules ] section ---
    try:
        molecules_section_start = -1
        molecules_section_end = -1
        for i, line in enumerate(lines):
            if re.match(r'^\s*\[\s*molecules\s*\]', line):
                molecules_section_start = i
            elif molecules_section_start != -1 and line.strip().startswith('['):
                molecules_section_end = i
                break
        
        if molecules_section_start == -1:
            # If no [molecules] section, add one at the end
            lines.append("\n[ molecules ]\n")
            lines.append("; Compound        #mols\n")
            molecules_section_start = len(lines) - 2

        if molecules_section_end == -1:
            molecules_section_end = len(lines)

        # Create the new molecules list
        new_molecule_lines = [lines[molecules_section_start], "; Compound        #mols\n"]
        for name, count in molecule_counts.items():
            if count > 0:
                new_molecule_lines.append(f"{name:<15} {count}\n")

        # Replace the old molecules section with the new one
        del lines[molecules_section_start+1:molecules_section_end]
        lines[molecules_section_start+1:molecules_section_start+1] = new_molecule_lines[1:]

        with open(topology_file, 'w') as f:
            f.writelines(lines)
        
        print(f"'{topology_file}'의 [ molecules ] 섹션을 성공적으로 업데이트했습니다.")

    except Exception as e:
        print(f"토폴로지 파일 업데이트 중 오류 발생: {e}", file=os.sys.stderr)
        raise
