import argparse

def modify_xyz_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()
        mol_id = 0
        i = 0
        
        while i < len(lines):
            # Ensure the atom count line is valid
            try:
                num_atoms = int(lines[i].strip())
            except ValueError:
                # Skip this molecule if we can't parse the atom count
                print(f"Warning: Skipping invalid line {i+1}: {lines[i].strip()}")
                i += 1
                continue

            # Write the number of atoms line
            outfile.write(lines[i])
            i += 1

            # Modify the second line with the molecule identifier
            if i < len(lines) and (lines[i].strip() == '' or lines[i].strip().lower() == 'optimized structure'):
                outfile.write(f"mol_id: {mol_id}\n")
            else:
                outfile.write(f"mol_id: {mol_id}\n")
            mol_id += 1
            i += 1
            
            # Write the remaining lines for this molecule
            for j in range(num_atoms):
                if i < len(lines):
                    outfile.write(lines[i])
                i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify XYZ file to add molecule IDs")
    parser.add_argument("input_file", type=str, help="Path to the input XYZ file")
    parser.add_argument("output_file", type=str, help="Path to the output XYZ file")

    args = parser.parse_args()
    modify_xyz_file(args.input_file, args.output_file)
