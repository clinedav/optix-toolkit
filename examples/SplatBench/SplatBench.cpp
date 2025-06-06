// SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <fstream>
#include <iterator>

// This include is needed to avoid a link error
#include <optix_stubs.h>

#include <OptiXToolkit/OTKAppBase/OTKApp.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include "SplatBenchParams.h"
#include "SplatLoader.h"

#include "SplatBenchKernelCuda.h"

using namespace otkApp;

struct SplatBenchPerDeviceExtraState
{
    void* d_esplats;
    void* d_hesplats;
    void* d_splats;
    void* d_aabbs;
};

//------------------------------------------------------------------------------
// SplatBenchApp
//------------------------------------------------------------------------------

class SplatBenchApp : public OTKApp
{
  public:
    SplatBenchApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop );
    void initView() override;
    void initLaunchParams( OTKAppPerDeviceOptixState& state, unsigned int numDevices ) override;
    void loadScene( std::string fileName );
    void buildAccel( OTKAppPerDeviceOptixState& state ) override;
    void setNumSplats( int numSplats ) { m_numSplats = numSplats; }
    void setSplatScale( float splatScale ) { m_splatScale = splatScale; }
   
  protected:
    int m_sceneId = 0;
    int m_numSplats = 1000;
    float m_splatScale = 1.0f;

    std::vector<EllipsoidSplat> m_esplats;
    std::vector<HalfSphereSplat> m_splats;
    std::vector<SplatBenchPerDeviceExtraState> m_perDeviceExtraStates;

    void sortSplats();
    void keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods ) override; 
};


SplatBenchApp::SplatBenchApp( const char* appTitle, unsigned int width, unsigned int height, const std::string& outFileName, bool glInterop )
    : OTKApp( appTitle, width, height, outFileName, glInterop, UI_ORBIT )
{
    m_backgroundColor = float4{1.0f, 0.0f, 0.0f, 0.0f};
    m_projection = Projection::PINHOLE;
    m_lens_width = 0.0f;
    m_render_mode = 1;

    m_perDeviceExtraStates.resize( m_perDeviceOptixStates.size() );
    for( unsigned int i=0; i<m_perDeviceOptixStates.size(); ++i )
    {
        m_perDeviceOptixStates[i].extraData = & m_perDeviceExtraStates[i];
    }
}

void SplatBenchApp::initView()
{
    setView( float3{-0.6f, -0.5f, 2.8f}, float3{-1.0f, -1.5f, 0.0f}, float3{0.0f, 1.0f, 0.0f}, 30.0f );
}

void SplatBenchApp::initLaunchParams( OTKAppPerDeviceOptixState& state, unsigned int numDevices )
{
    OTKApp::initLaunchParams( state, numDevices );
    SplatBenchPerDeviceExtraState* extraState = reinterpret_cast<SplatBenchPerDeviceExtraState*>( state.extraData );

    // SplatBench specific data
    SplatBenchParams* params = reinterpret_cast<SplatBenchParams*>( &state.params.extraData );
    params->s_splats = (HalfSphereSplat*)extraState->d_splats;
    params->e_splats = (EllipsoidSplat*)extraState->d_esplats;
    params->splatScale = m_splatScale;
}

void SplatBenchApp::loadScene( std::string modelFileName )
{
    // Dummy material
    m_materials.emplace_back();

#ifndef USE_COOP_VEC
    // No model. Make a grid of splats.
    if( modelFileName == "" )
    {
        EllipsoidSplat ep;
        ep.Q = float3{0.0f, 0.0f, 0.0f};
        ep.A = float3{0.5f, 0.0f, 0.0f};
        ep.B = float3{0.0f, 0.5f, 0.0f};
        ep.C = float3{0.0f, 0.0f, 0.5f};
        ep.color = uchar4{255, 255, 255, 32};
        m_esplats.push_back(ep);
    }
    
    else
    {
        GaussianSplatPlyLoader gs;
        gs.load( modelFileName.c_str() );
        printf("Loaded %s.  (%d points)\n", modelFileName.c_str(), gs.getNumSplats());
        unsigned int numSplats = ( m_numSplats <= 0 ) ? gs.getNumSplats() : std::min((unsigned int)m_numSplats, gs.getNumSplats());
        for( unsigned int i=0; i<numSplats; ++i )
        {
            float3 flip = float3{1.0f, -1.0f, -1.0f};
            float3 center = gs.center(i);
            center = center * flip;
            float3 scale = gs.scale(i);
            float4 color = SRGBAtoRGBA( gs.color(i) );
           
            EllipsoidSplat ep;
            ep.Q = center;
            
            float4 rotation = normalize( gs.rotation(i) );
            float3 A, B, C;
            getBasisFromNormalizedQuaternion( rotation, A, B, C );
            A = A * flip;
            B = B * flip;
            C = C * flip;

            ep.A = ( A * ( m_splatScale/scale.x ) );
            ep.B = ( B * ( m_splatScale/scale.y ) );
            ep.C = ( C * ( m_splatScale/scale.z ) );
            color *= 255.0f;
            ep.color = uchar4{(uchar)color.x, (uchar)color.y, (uchar)color.z, (uchar)color.w};
            
            //ep.color.x = std::min(ep.color.x+3, 255) & 0xfa;
            //ep.color.y = std::min(ep.color.y+3, 255) & 0xfa;
            //ep.color.z = std::min(ep.color.z+3, 255) & 0xfa;
            //ep.color.w = std::min(ep.color.w+3, 255) & 0xfa;

            m_esplats.push_back(ep);
        }
        m_splatScale = 1.0f;
        sortSplats();
    }
#endif
}


float3 esmin = float3{1e7f, 1e7f, 1e7f};
float3 esmax = float3{-1e7f, -1e7f, -1e7f};

unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

unsigned int morton3D(float x, float y, float z)
{
    x = std::min(std::max(x * 1024.0f, 0.0f), 1023.0f);
    y = std::min(std::max(y * 1024.0f, 0.0f), 1023.0f);
    z = std::min(std::max(z * 1024.0f, 0.0f), 1023.0f);
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
}

bool mortonEsplatCompare( const EllipsoidSplat& a, const EllipsoidSplat& b )
{
    float ax = (a.Q.x - esmin.x) / (esmax.x - esmin.x);
    float ay = (a.Q.y - esmin.y) / (esmax.y - esmin.y);
    float az = (a.Q.z - esmin.z) / (esmax.z - esmin.z);
    uint32_t acode = morton3D( ax, ay, az );

    float bx = (b.Q.x - esmin.x) / (esmax.x - esmin.x);
    float by = (b.Q.y - esmin.y) / (esmax.y - esmin.y);
    float bz = (b.Q.z - esmin.z) / (esmax.z - esmin.z);
    uint32_t bcode = morton3D( bx, by, bz );

    return acode <= bcode;
}

void SplatBenchApp::sortSplats()
{
    for( EllipsoidSplat& s : m_esplats )
    {
        esmin.x = std::min( esmin.x, s.Q.x );
        esmin.y = std::min( esmin.y, s.Q.y );
        esmin.z = std::min( esmin.z, s.Q.z );

        esmax.x = std::max( esmax.x, s.Q.x );
        esmax.y = std::max( esmax.y, s.Q.y );
        esmax.z = std::max( esmax.z, s.Q.z );
    }
    std::sort( m_esplats.begin(), m_esplats.end(), mortonEsplatCompare );
}

OptixAabb expandBox( const OptixAabb& a, const OptixAabb& b )
{
    OptixAabb c;
    c.minX = std::min( a.minX, b.minX );
    c.minY = std::min( a.minY, b.minY );
    c.minZ = std::min( a.minZ, b.minZ );

    c.maxX = std::max( a.maxX, b.maxX );
    c.maxY = std::max( a.maxY, b.maxY );
    c.maxZ = std::max( a.maxZ, b.maxZ );
    return c;
}


void SplatBenchApp::buildAccel( OTKAppPerDeviceOptixState& state )
{
    //FIXME: Handle custom primitives, triangles, lss based on mode

    // Copy splats data to device
    SplatBenchPerDeviceExtraState* extraStateData = (SplatBenchPerDeviceExtraState*)state.extraData;
    extraStateData->d_esplats = cudaMallocAndCopyToDevice( m_esplats.data(), m_esplats.size() * sizeof(EllipsoidSplat) );

    // Make AABBs and copy to device
    
    std::vector<OptixAabb> boxes( m_esplats.size() );
    for( unsigned int i=0; i<boxes.size(); ++i )
    {
        boxes[i] = getAabb( m_esplats[i] );
    }
    

    // Make AABBs and copy to device for groups of boxes
    /*
    std::vector<OptixAabb> boxes( m_esplats.size()/32 );
    for( unsigned int i=0; i<boxes.size(); ++i )
    {
        boxes[i] = getAabb( m_esplats[i*32] );
        for( int j=0; j<32; ++j )
            boxes[i] = expandBox( boxes[i], getAabb( m_esplats[i*32+j] ) );
    }
    */

    extraStateData->d_aabbs = cudaMallocAndCopyToDevice( boxes.data(), boxes.size() * sizeof(OptixAabb) );

    // Copy vertex and material data to device
    state.d_vertices = (CUdeviceptr) cudaMallocAndCopyToDevice( m_vertices.data(), m_vertices.size() * sizeof(float4) );
    uint32_t* d_material_indices = (uint32_t*) cudaMallocAndCopyToDevice( m_material_indices.data(), m_material_indices.size() * sizeof( uint32_t ) );

    std::vector<uint32_t> aabb_input_flags( m_materials.size(), m_optixGeometryFlags );
    OptixBuildInput aabb_input = {};
    aabb_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    aabb_input.customPrimitiveArray.aabbBuffers   = (CUdeviceptr*)&extraStateData->d_aabbs;
    aabb_input.customPrimitiveArray.flags         = aabb_input_flags.data();
	aabb_input.customPrimitiveArray.numSbtRecords = 1;
    aabb_input.customPrimitiveArray.numPrimitives = (unsigned int)boxes.size(); //m_esplats.size();
    aabb_input.customPrimitiveArray.sbtIndexOffsetBuffer      = (CUdeviceptr)d_material_indices;
    aabb_input.customPrimitiveArray.sbtIndexOffsetSizeInBytes = sizeof( uint32_t );

    // Accel options
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation              = OPTIX_BUILD_OPERATION_BUILD;

    // Compute memory usage for accel build
    OptixAccelBufferSizes gas_buffer_sizes;
    const unsigned int num_build_inputs = 1;
    OTK_ERROR_CHECK( optixAccelComputeMemoryUsage( state.context, &accel_options, &aabb_input, num_build_inputs, &gas_buffer_sizes ) );

    // Allocate temporary buffer needed for accel build
    void* d_temp_buffer = nullptr;
    OTK_ERROR_CHECK( cudaMalloc( &d_temp_buffer, gas_buffer_sizes.tempSizeInBytes ) );

    // Allocate output buffer for (non-compacted) accel build result, and also compactedSize property.
    void* d_buffer_temp_output_gas_and_compacted_size = nullptr;
    size_t      compactedSizeOffset = roundUp<size_t>( gas_buffer_sizes.outputSizeInBytes, 8ull );
    OTK_ERROR_CHECK( cudaMalloc( &d_buffer_temp_output_gas_and_compacted_size, compactedSizeOffset + 8 ) );

    // Set up the accel build to return the compacted size, so compaction can be run after the build
    OptixAccelEmitDesc emitProperty = {};
    emitProperty.type               = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitProperty.result             = ( CUdeviceptr )( (char*)d_buffer_temp_output_gas_and_compacted_size + compactedSizeOffset );

    // Finally perform the accel build
    OTK_ERROR_CHECK( optixAccelBuild(
                state.context,
                CUstream{0},
                &accel_options,
                &aabb_input,
                num_build_inputs,
                reinterpret_cast<CUdeviceptr>( d_temp_buffer ),
                gas_buffer_sizes.tempSizeInBytes,
                reinterpret_cast<CUdeviceptr>( d_buffer_temp_output_gas_and_compacted_size ),
                gas_buffer_sizes.outputSizeInBytes,
                &state.gas_handle,
                &emitProperty,
                1
                ) );

    // Delete temporary buffers used for the accel build
    OTK_ERROR_CHECK( cudaFree( d_temp_buffer ) );
    OTK_ERROR_CHECK( cudaFree( d_material_indices ) );

    // Copy the size of the compacted GAS accel back from the device
    size_t compacted_gas_size;
    OTK_ERROR_CHECK( cudaMemcpy( &compacted_gas_size, (void*)emitProperty.result, sizeof(size_t), cudaMemcpyDeviceToHost ) );

    // If compaction reduces the size of the accel, copy to a new buffer and delete the old one
    if( compacted_gas_size < gas_buffer_sizes.outputSizeInBytes )
    {
        OTK_ERROR_CHECK( cudaMalloc( reinterpret_cast<void**>( &state.d_gas_output_buffer ), compacted_gas_size ) );
        // use handle as input and output
        OTK_ERROR_CHECK( optixAccelCompact( state.context, 0, state.gas_handle, state.d_gas_output_buffer, compacted_gas_size, &state.gas_handle ) );
        OTK_ERROR_CHECK( cudaFree( (void*)d_buffer_temp_output_gas_and_compacted_size ) );
    }
    else
    {
        state.d_gas_output_buffer = reinterpret_cast<CUdeviceptr>( d_buffer_temp_output_gas_and_compacted_size );
    }
}


void SplatBenchApp::keyCallback( GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods )
{
    OTKApp::keyCallback( window, key, scancode, action, mods );
    if( action != GLFW_PRESS )
        return;

    if( key == GLFW_KEY_J )
        m_splatScale *= 0.95f;
    else if( key == GLFW_KEY_L )
        m_splatScale *= 1.05f;
    else if( key == GLFW_KEY_K )
        m_splatScale = 1.0f;
    printf("splatScale: %1.2f\n", m_splatScale);

    m_subframeId = 0;
}


//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------

void printUsageAndExit( const char* argv0 )
{
    std::cerr << "\n\nUsage: " << argv0 << " [options]\n\n";
    std::cout << "Options:  --launches <numLaunches> --dim=<width>x<height>, --file <outputfile.ppm>, --no-gl-interop\n\n";
    std::cout << "Mouse:    <LMB>:          pan camera\n";
    std::cout << "          <RMB>:          rotate camera\n\n";
    std::cout << "Keyboard: <ESC>:          exit\n";
    std::cout << "          WASD,QE:        pan camera\n";
    std::cout << "          J,L:            rotate camera\n";
    std::cout << "          C:              reset view\n\n";
}

int main( int argc, char* argv[] )
{
    int         windowWidth   = 1280;
    int         windowHeight  = 1024;
    bool        glInterop     = true;
    int         numLaunches   = 32;
    int         numSplats  = -1;
    float       splatScale = 1.0f;
    std::string modelFileName = "";
    std::string outFileName   = "";

    for( int i = 1; i < argc; ++i )
    {
        const std::string arg( argv[i] );
        const bool        lastArg = ( i == argc - 1 );

        if( ( arg == "--file" ) && !lastArg )
            outFileName = argv[++i];
        else if( arg.substr( 0, 6 ) == "--dim=" )
            otk::parseDimensions( arg.substr( 6 ).c_str(), windowWidth, windowHeight );
        else if( arg == "--no-gl-interop" )
            glInterop = false;
        else if( arg == "--model" && !lastArg )
            modelFileName = argv[++i];
        else if( arg == "--num-splats" && !lastArg )
            numSplats = atoi( argv[++i] );
        else if( arg == "--splat-scale" && !lastArg )
            splatScale = atof( argv[++i] );
        else 
            printUsageAndExit( argv[0] );
    }
    if( numSplats < 0 && modelFileName == "" )
        numSplats = 1000;

    SplatBenchApp app( "Splat Viewer", windowWidth, windowHeight, outFileName, glInterop );

    app.setRaygenProgram( "__raygen__splatBench" );
    app.setMissProgram( nullptr );
    app.setAnyhitProgram( nullptr ); 
    app.setClosestHitProgram( nullptr );
    app.setIntersectionProgram( "__intersection__ellipsoidSplat" );

    app.initView();
    app.setNumSplats( numSplats );
    app.setNumLaunches( numLaunches );
    app.setSplatScale( splatScale );
    app.loadScene( modelFileName );
    app.resetAccumulator();
    app.initOptixPipelines( SplatBenchCudaText(), SplatBenchCudaSize );
    app.startLaunchLoop();
    
    return 0;
}
