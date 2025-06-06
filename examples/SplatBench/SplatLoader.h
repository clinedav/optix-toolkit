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

/*
0.5 + 0.28209479177387814 * RGB

Position (x, y, z) from PLY 
Position in UE = (x, -z, -y) * 100.0f

Scale (x, y, z) from PLY
Scale in UE = (1/1+exp(-x), 1/1+exp(-y), 1/1+exp(-z)) * 100.0f

Orientation (w, x, y, z) from PLY
Orientation in UE = normalized(x, y, z, w)

Opacity (x) from PLY
Opacity in UE = 1 / 1 + exp(-x)
*/

#pragma once

struct GSIndex
{
    int x, y, z;
    int r, g, b, opacity;
    int s0, s1, s2;
    int r0, r1, r2, r3;
};

/*
class PlyLoader
{
  public:
    bool tokenizeLine( std::ifstream& file, std::vector<std::string> tokens )
    {
        std::string line;
        std::getline( file, line );
        tokens.clear();
        std::istringstream iss( line );
        while( iss.good() )
        {
            std::string token;
            iss >> token;
            tokens.push_back(token);
        }
    }

    bool load( const char* fileName )
    {
        // Open file and get its size
        std::ifstream file( fileName, std::ios::binary );
        if( !file.is_open() ) 
            return false;
        file.unsetf( std::ios::skipws );

        // Read magic number
        std::vector<string> tokens;
        tokenizeLine( file, tokens );
        if( tokens[0] != "ply" )
        {
            file.close();
            return false;
        }

        // Read header
        while( tokenizeLine( file, tokens ) )
        {
            if( tokens[0] == "format" && tokens[1] != "binary_little_endian" )
            {
                file.close();
                return false;
            }
            else if( tokens[0] == "element" )
            {
                elements.add( Element(tokens) );
            }
            else if( tokens[0] == "property" )
            {
                elements.back().properties.add( Property(tokens) );
            }
            else if( tokens[] == "end_header" )
            {
                break;
            }
        }

        // Read the rest of the file into a buffer
        std::copy(std::istream_iterator<char>(file), std::istream_iterator<char>(), std::back_inserter(m_data));
        file.close();

        for(unsigned int i=0; i<elements.size(); ++i)
        {
            printf("%s, %s\n", elements[i].name, elements[i].size);
        }
        return true;
    }

    struct PlyProperty
    {
        std::string name;
        std::string type;
        int size;
    };
    struct PlyElement
    {
        std::string name;
        std::vector<Property> properties;
        int size;
    }

    std::vector<PlyElement> elements;
    std::vector<char> data;
};
*/

class GaussianSplatPlyLoader
{
  public:
    float3 center(int i)
    {
        float* p = (float*)&m_data[i*sizeof(float)*m_numSplatProps];
        return float3{p[m_idx.x], p[m_idx.y], p[m_idx.z]};
    }
    float4 rotation(int i)
    {
        float* p = (float*)&m_data[i*sizeof(float)*m_numSplatProps];
        return float4{p[m_idx.r1], p[m_idx.r2], p[m_idx.r3], p[m_idx.r0]};
    }
    float3 scale(int i)
    {
        float* p = (float*)&m_data[i*sizeof(float)*m_numSplatProps];
        float s0 = 1.0f / (1.0f + expf(-p[m_idx.s0]));
        float s1 = 1.0f / (1.0f + expf(-p[m_idx.s1]));
        float s2 = 1.0f / (1.0f + expf(-p[m_idx.s2]));
        return float3{s0, s1, s2};
    }
    float4 color(int i)
    {
        const float c0 = 0.28209479f;
        float* p = (float*)&m_data[i*sizeof(float)*m_numSplatProps];
        float r = std::max( 0.5f + p[m_idx.r] * c0, 0.0f );
        float g = std::max( 0.5f + p[m_idx.g] * c0, 0.0f );
        float b = std::max( 0.5f + p[m_idx.b] * c0, 0.0f );
        float a = 1.0f / (1.0f + expf( -p[m_idx.opacity] ));
        return float4{r, g, b, a};
    }
    unsigned int getNumSplats() { return m_numSplats; }

    bool load( const char* fileName )
    {
        // Open file and get its size
        std::ifstream file( fileName, std::ios::binary );
        if( !file.is_open() ) 
            return false;
        file.unsetf(std::ios::skipws);

        // Reset properties
        m_numSplatProps = 0;
        m_numSplats = 0;
        m_idx = GSIndex{};

        // Read file header
        std::string line;
        while ( std::getline( file, line ) ) 
        {
            std::istringstream iss( line );
            std::string token;
            iss >> token;
            if( token == "format" ) 
            {
                std::string format; 
                iss >> format;
                if( format != "binary_little_endian" ) // expecting binary format
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