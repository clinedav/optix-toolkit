// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

/*
ply
format binary_little_endian 1.0
element vertex 559263
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
property float f_rest_1
...
property float f_rest_44
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
*/

#pragma once

class PlyLoader
{
  public:
    bool load( const char* fileName )
    {
        // Open file and get its size
        std::ifstream file( fileName, std::ios::binary );
        if( !file.is_open() ) 
            return false;
        file.unsetf( std::ios::skipws );

        // Read magic number
        std::string line;
        std::getline( file, line );
        if( line != "ply" )
            return false;

        // Read header
        while ( std::getline( file, line ) ) 
        {
            std::istringstream iss( line );
            std::string attribute;
            iss >> token;
            if( token == "format" ) 
            {
                std::string format; 
                iss >> format;
                if( format != "binary_little_endian" ) // only supporting this format for now
                    return false;
            } 
            else if( token == "element" ) 
            {
                std::string elementName;
                iss >> elementName;
                if( elementName != "vertex" ) // splats stored as vertices
                    return false;
                iss >> m_numSplats;
            } 
            else if( token == "property" ) 
            {
                std::string propName;
                iss >> propName; // skip type, assuming float
                iss >> propName;
                addIndex( propName, m_numSplatProps );
                m_numSplatProps++;
            } 
            else if (token == "end_header") 
            {
                break;
            }
        }

        if( m_numSplats <= 0 || m_numSplatProps <= 0 )
            return false;

        // Read the reset of the file into a buffer
        std::copy(std::istream_iterator<char>(file), std::istream_iterator<char>(), std::back_inserter(m_data));
        
        file.close();
        return true;
    }

  protected:
    void addIndex( std::string prop, int offset )
    {
        if( prop == "x" ) m_idx.x = offset;
        else if( prop == "y" ) m_idx.y = offset;
        else if( prop == "z" ) m_idx.z = offset;
        else if( prop == "f_dc_0" ) m_idx.r = offset;
        else if( prop == "f_dc_1" ) m_idx.g = offset;
        else if( prop == "f_dc_2" ) m_idx.b = offset;
        else if( prop == "opacity" ) m_idx.opacity = offset;
        else if( prop == "scale_0" ) m_idx.s0 = offset;
        else if( prop == "scale_1" ) m_idx.s1 = offset;
        else if( prop == "scale_2" ) m_idx.s2 = offset;
        else if( prop == "rot_0" ) m_idx.r0 = offset;
        else if( prop == "rot_1" ) m_idx.r1 = offset;
        else if( prop == "rot_2" ) m_idx.r2 = offset;
        else if( prop == "rot_3" ) m_idx.r3 = offset;
    }

    int m_numSplats;
    int m_numSplatProps;
    GSIndex m_idx;
    std::vector<char> m_data;
};