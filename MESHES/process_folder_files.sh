#!/bin/bash

# Step 1: Navigate to the folder
folder_name=$1
if [ -z "$folder_name" ]; then
    echo "Usage: $0 folder_name"
    exit 1
fi
cd "$folder_name" || { echo "Folder not found: $folder_name"; exit 1; }

# Step 2: Replace the finished .dat file
dat_file="${folder_name}.dat"
cat > "$dat_file" << EOF
$-------------------------------------------------------------------
RUN_DATA
  ALYA:                   $folder_name
END_RUN_DATA
$-------------------------------------------------------------------
PROBLEM_DATA
  TIME_COUPLING:          Global, From_ritical
  TIME_INTERVAL=          0.0, 1.0  
  TIME_STEP_SIZE=         1.0  
  NUMBER_OF_STEPS=        1
  EQUILI_MODULE:          On
  END_EQUILI_MODULE
  PARALL_SERVICE:         Off
    PARTITION_TYPE:     FACES
  END_PARALL_SERVICE
END_PROBLEM_DATA
$-------------------------------------------------------------------
EOF

# Correct separators
sed -i 's/hB------------------------------------------------------------------/$-------------------------------------------------------------------/g' "$dat_file"

# Step 3: Set $Elemtype based on folder_name
if [[ "$folder_name" == *"TRI03"* ]]; then
    Elemtype=10
elif [[ "$folder_name" == *"TRI06"* ]]; then
    Elemtype=11
elif [[ "$folder_name" == *"TRI10"* ]]; then
    Elemtype=16
elif [[ "$folder_name" == *"QUA04"* ]]; then
    Elemtype=12
elif [[ "$folder_name" == *"QUA09"* ]]; then
    Elemtype=14
elif [[ "$folder_name" == *"QUA16"* ]]; then
    Elemtype=15
else
    echo "Unknown element type in folder_name."
    exit 1
fi

# Step 4: Modify the .dom.dat file
dom_file="${folder_name}.dom.dat"
if [ -f "$dom_file" ]; then
    # 1. Replace the TYPES_OF_ELEMENTS line with the new $Elemtype
    sed -i "s/TYPES_OF_ELEMENTS=.*/TYPES_OF_ELEMENTS=      $Elemtype/" "$dom_file"

    # 2. Replace lines between GEOMETRY and END_BOUNDARY_CONDITIONS, removing any duplicates
    awk -v folder="$folder_name" '
    BEGIN {inside_geometry = 0}
    /GEOMETRY/ && inside_geometry == 0 {
        inside_geometry = 1
        print "GEOMETRY\n  INCLUDE  " folder ".geo.dat\nEND_GEOMETRY\n$-------------------------------------------------------------\nSETS\n  ELEME, ALL = 10\n  END_ELEME\nEND_SETS\n$-------------------------------------------------------------\nBOUNDARY_CONDITIONS\n  INCLUDE  " folder ".fix.dat\nEND_BOUNDARY_CONDITIONS\n$-------------------------------------------------------------"
        next
    }
    inside_geometry == 0 {print}
    /END_BOUNDARY_CONDITIONS/ {inside_geometry = 0}
    ' "$dom_file" > temp_dom_file && mv temp_dom_file "$dom_file"
    
    # 3. Erase everything after "END_BOUNDARY_CONDITIONS" and retain the separator
    sed -i '/END_BOUNDARY_CONDITIONS/{n;/^$/!q}' "$dom_file"
else
    echo "File not found: $dom_file"
fi

# Step 5: Modify the .geo.dat file
geo_file="${folder_name}.geo.dat"
if [ -f "$geo_file" ]; then
    # Remove everything between MATERIALS and END_MATERIALS and replace
    sed -i '/MATERIALS/,/END_MATERIALS/c\  MATERIALS, NUMBER= 1, DEFAULT=1\n  END_MATERIALS' "$geo_file"
else
    echo "File not found: $geo_file"
fi

echo "Finished processing $folder_name"
