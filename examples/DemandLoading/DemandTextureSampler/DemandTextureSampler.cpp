// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <cuda.h>

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include "DemandLoaderImpl.h"
#include "Memory/DeviceMemoryManager.h"
#include "PageTableManager.h"
#include "Textures/DemandTextureImpl.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>
#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Error/optixErrorCheck.h>

#include <OptiXToolkit/DemandLoading/DemandLoader.h>
#include <OptiXToolkit/DemandLoading/DemandTexture.h>
#include <OptiXToolkit/DemandLoading/SparseTextureDevices.h>
#include <OptiXToolkit/DemandLoading/TextureDescriptor.h>
#include <OptiXToolkit/DemandLoading/MultiCheckerImage.h>

#include <DemandTextureSamplerKernelCuda.h>

#include "DemandTextureSampler.h"

using namespace demandLoading;
using namespace imageSource;
using namespace otk;

using ImageSourcePtr = std::shared_ptr<imageSource::ImageSource>;

//
// Shader binding table definitions
//

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<int> RayGenSbtRecord;


//
// Texture function names
//

const std::vector<std::string> textureFunctionNames = {
    "tex2D",
    "tex2DLod",
    "tex2DGrad",
    "tex2DGradUdim",
    "tex2DGradUdimBlend",
    "tex2DCubic",
    "tex2DCubicDerivatives",
    "tex2DCubicUdim",
    "tex2DCubicUdimDerivatives",
    "ntcTex2D",
    "ntcTex2DGrad"
};


//
// Per device Optix state
//

struct OptixState
{
    CUstream                    stream              = 0;   // Cuda stream where OptiX launches will occur
    OptixDeviceContext          optixContext        = 0;   // OptiX context for this device
    OptixModule                 optixir_module      = 0;   // OptiX module stores device code from a file
    OptixPipeline               pipeline            = 0;   // One or more modules linked together for device side programs of the app

    OptixProgramGroup           raygen_prog_group   = 0;   // OptiX raygen programs for the pipeline
    OptixShaderBindingTable     sbt                 = {};  // Shader binding table

    std::vector<TextureSample>  textureSamples;
    CUdeviceptr                 d_textureSamples    = 0;   // Device-side copy of texture samples
    DemandTextureSamplerLaunchParams params         = {};  // Host-side copy of parameters for OptiX launch
    CUdeviceptr                 d_params            = 0;   // Device-side copy of params

    std::shared_ptr<demandLoading::DemandLoader> demandLoader;  // Manage demand load requests
    demandLoading::Ticket ticket;                               // Track demand load requests for last OptiX launch
};


//
// DemandTextureSampler class
//

class DemandTextureSampler
{
  public:
    DemandTextureSampler() {}
    void printUsage();
    bool processCommand( const std::string& command );
    void shutdown();

  protected:
    demandLoading::Options options{};
    demandLoading::TextureDescriptor m_textureDesc{};
    std::map<std::string, int> m_textureIdMap{};
    OptixState m_optixState{};

    void splitString( const std::string& line, std::vector<std::string>& tokens );
    bool init();
    bool isInitialized() const { return m_optixState.optixContext != 0;}
    bool loadTexture( std::vector<std::string>& tokens );
    bool launch( int numLaunches = 1 );
    void printSample( TextureSample& sample );
};


//
// DemandTextureSampler implementation
//

void DemandTextureSampler::printUsage()
{
    std::cerr << "\nGeneral Commands:\n"
        "    help\n"
        "    launch [<numLaunches>]\n"
        "    quit\n"
        "\nDemand loading options: (must be specified before first loadTexture command)\n"
        "    maxRequestedPages <numPages>\n"
        "    useSparseTextures <true|false>\n"
        "    useCascadingTextureSizes <true|false>\n"
        "    coalesceWhiteBlackTiles <true|false>\n"
        "    coalesceDuplicateImages <true|false>\n"
        "\nTexture Loading Commands:\n"
        "    addressMode <border|clamp|wrap> <border|clamp|wrap>\n"
        "    filterMode <point|linear|bicubic|smartbicubic>\n"
        "    mipmapFilterMode <point|linear>\n"
        "    maxAnisotropy <1-16>\n"
        "    conservativeFilter <true|false>\n"
        "    loadTexture <textureId> <textureName>\n"
        "    loadTexture <textureId> checkerboard <type> <width> <height> [<mipmap>]\n"
        "        (type must be one of: uchar, uchar2, uchar4, half, half2, half4, float, float2, float4)\n"
        "\nTexture sample commands:\n"
        "    clearSamples\n"
        "    tex2D <textureId> <u> <v> [<jitter>]\n"
        "    tex2DLod <textureId> <u> <v> <lod> [<jitter>]\n"
        "    tex2DGrad <textureId> <u> <v> <ddx> <ddy> [<jitter>]\n"
        "    tex2DGradUdim <textureId> <u> <v> <ddx> <ddy> [<jitter>]\n"
        "    tex2DGradUdimBlend <textureId> <u> <v> <ddx> <ddy> [<jitter>]\n"
        "    tex2DCubic <textureId> <u> <v> <ddx> <ddy> [<jitter>]\n"
        "    tex2DCubicDerivatives <textureId> <u> <v> <ddx> <ddy> [<jitter>]\n"
        "    tex2DCubicUdim <textureId> <u> <v> <ddx> <ddy> [<jitter>]\n"
        "    tex2DCubicUdimDerivatives <textureId> <u> <v> <ddx> <ddy> [<jitter>]\n"
#ifdef INCLUDE_NTC_TEXTURES
        "    ntcTex2DGrad <textureId> <u> <v> <ddx> <ddy> [<jitter>]\n"
        "    ntcTex2DGradUdim <textureId> <u> <v> <ddx> <ddy> [<jitter>]\n"
#endif
        "\n";
}

bool DemandTextureSampler::processCommand( std::string line )
{
    // Split the line into tokens
    std::vector<std::string> tokens;
    splitString( line, tokens );

    // empty command and comments
    if( tokens.size() == 0 || command[0] == '#' || command[0] == '/' )
        return true;

    const std::string& command = tokens[0];
    TextureSample sample{};

    // General commands
    if( command == "help" )
    {
        printUsage( "" );
        return true;
    }
    else if( command == "launch" )
    {
        int numLaunches = (tokens.size() > 1) ? std::stoi( tokens[1] ) : 1;
        for( int i = 0; i < numLaunches; i++ )
        {
            if( !launch() )
                return false;
        }
        return true;
    }
    else if( command == "quit" )
    {
        return false;
    }

    // Demand loading options
    else if( command == "maxRequestedPages" )
    {
        m_options.maxRequestedPages = std::stoi( tokens[1] );
        return true;
    }
    else if( command == "useSparseTextures" )
    {
        m_options.useSparseTextures = !(tokens[1][0] == 'f' || tokens[1][0] == 'F');
        return true;
    }
    else if( command == "useCascadingTextureSizes" )
    {
        m_options.useCascadingTextureSizes = !(tokens[1][0] == 'f' || tokens[1][0] == 'F');
        return true;
    }
    else if( command == "coalesceWhiteBlackTiles" )
    {
        m_options.coalesceWhiteBlackTiles = !(tokens[1][0] == 'f' || tokens[1][0] == 'F');
        return true;
    }
    else if( command == "coalesceDuplicateImages" )
    {
        m_options.coalesceDuplicateImages = !(tokens[1][0] == 'f' || tokens[1][0] == 'F');
        return true;
    }

    // Texture loading commands
    else if( command == "addressMode" )
    {
        if( tokens.size() < 3 )
        {
            std::cerr << "Error: addressMode requires 3 arguments" << std::endl;
            return false;
        }

        if( tokens[1] == "border" ) m_textureDesc.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
        if( tokens[1] == "clamp" ) m_textureDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        if( tokens[1] == "wrap" ) m_textureDesc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
        
        if( tokens[2] == "border" ) m_textureDesc.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
        if( tokens[2] == "clamp" ) m_textureDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        if( tokens[2] == "wrap" ) m_textureDesc.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;

        return true;
    }
    else if( command == "filterMode" )
    {
        if( tokens[1] == "point" ) m_textureDesc.filterMode = FILTER_POINT;
        if( tokens[1] == "linear" ) m_textureDesc.filterMode = FILTER_LINEAR;
        if( tokens[1] == "bicubic" ) m_textureDesc.filterMode = FILTER_BICUBIC;
        if( tokens[1] == "smartbicubic" ) m_textureDesc.filterMode = FILTER_SMARTBICUBIC;
        return true;
    }
    else if( command == "mipmapFilterMode" )
    {
        if( tokens[1] == "point" ) m_textureDesc.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
        if( tokens[1] == "linear" ) m_textureDesc.mipmapFilterMode = CU_TR_FILTER_MODE_LINEAR;
        return true;
    }
    else if( command == "maxAnisotropy" )
    {
        m_textureDesc.maxAnisotropy = std::stoi( tokens[1] );
        return true;
    }
    else if( command == "conservativeFilter" )
    {
        m_textureDesc.conservativeFilter = !(tokens[1][0] == 'f' || tokens[1][0] == 'F');
        return true;
    }
    else if( command == "loadTexture" )
    {
        return loadTexture( tokens );
    }

    // Texture sample commands
    else if( command == "clearSamples" )
    {
        m_optixState.textureSamples.clear();
        return true;
    }
    else if( command == "tex2D" )
    {
        if( tokens.size() < 4 )
        {
            std::cerr << "Error: tex2D requires 4 arguments" << std::endl;
            return false;
        }
        sample.textureFunction = TF_tex2D;
        sample.textureId = getTextureId( tokens[1] );
        sample.u = std::stof( tokens[2] );
        sample.v = std::stof( tokens[3] );
        if( tokens.size() >= 6 )
            sample.jitter = float2{ std::stof( tokens[4] ), std::stof( tokens[5] ) };
        m_optixState.textureSamples.push_back( sample );
        return true;
    }
    else if( command == "tex2DLod" )
    {
        if( tokens.size() < 5 )
        {
            std::cerr << "Error: tex2DLod requires 5 arguments" << std::endl;
            return false;
        }
        sample.textureFunction = TF_tex2DLod;
        sample.textureId = getTextureId( tokens[1] );
        sample.u = std::stof( tokens[2] );
        sample.v = std::stof( tokens[3] );
        sample.lod = std::stof( tokens[4] );
        if( tokens.size() >= 7 )
            sample.jitter = float2{ std::stof( tokens[5] ), std::stof( tokens[6] ) };
        m_optixState.textureSamples.push_back( sample );
        return true;
    }
    else if( command == "tex2DGrad" || command == "tex2DGradUdim" || command == "tex2DGradUdimBlend"
        || command == "tex2DCubic" || command == "tex2DCubicDerivatives" 
        || command == "tex2DCubicUdim" || command == "tex2DCubicUdimDerivatives" )
    {
        if( tokens.size() < 8 )
        {
            std::cerr << "Error: tex2DGrad* requires 8 arguments" << std::endl;
            return false;
        }

        // Determine the texture function
        if( command == "tex2DGrad" )
            sample.textureFunction = TF_tex2DGrad;
        else if( command == "tex2DGradUdim" )
            sample.textureFunction = TF_tex2DGradUdim;
        else if( command == "tex2DGradUdimBlend" )
            sample.textureFunction = TF_tex2DGradUdimBlend;
        else if( command == "tex2DCubic" )
            sample.textureFunction = TF_tex2DCubic;
        else if( command == "tex2DCubicDerivatives" )
            sample.textureFunction = TF_tex2DCubicDerivatives;
        else if( command == "tex2DCubicUdim" )
            sample.textureFunction = TF_tex2DCubicUdim;
        else if( command == "tex2DCubicUdimDerivatives" )
            sample.textureFunction = TF_tex2DCubicUdimDerivatives;

        sample.textureFunction = TF_tex2DGrad;
        sample.textureId = getTextureId( tokens[1] );
        sample.u = std::stof( tokens[2] );
        sample.v = std::stof( tokens[3] );
        sample.ddx = float2{ std::stof( tokens[4] ), std::stof( tokens[5] ) };
        sample.ddy = float2{ std::stof( tokens[6] ), std::stof( tokens[7] ) };
        if( tokens.size() >= 9 )
            sample.jitter = float2{ std::stof( tokens[8] ), std::stof( tokens[9] ) };
        m_optixState.textureSamples.push_back( sample );
        return true;
    }

#ifdef INCLUDE_NTC_TEXTURES
    else if( command == "ntcTex2DGrad" || command == "ntcTex2DGradUdim" )
    {
        if( tokens.size() < 8 )
        {
            std::cerr << "Error: ntcTex2DGrad* requires 8 arguments" << std::endl;
            return false;
        }

        // Determine the texture function
        if( command == "ntcTex2DGrad" )
            sample.textureFunction = TF_ntcTex2DGrad;
        else if( command == "ntcTex2DGradUdim" )
            sample.textureFunction = TF_ntcTex2DGradUdim;

        sample.textureFunction = TF_ntcTex2DGradUdim;
        sample.textureId = getTextureId( tokens[1] );
        sample.u = std::stof( tokens[2] );
        sample.v = std::stof( tokens[3] );
        sample.ddx = float2{ std::stof( tokens[4] ), std::stof( tokens[5] ) };
        sample.ddy = float2{ std::stof( tokens[6] ), std::stof( tokens[7] ) };
        if( tokens.size() >= 9 )
            sample.jitter = float2{ std::stof( tokens[8] ), std::stof( tokens[9] ) };
        else
            sample.jitter = float2{ 0.5f, 0.5f };
        m_optixState.textureSamples.push_back( sample );
        return true;
    }
#endif

    // Unknown command
    else
    {
        std::cerr << "Unknown command: " << line << std::endl;
        printUsage();
        return false;
    }
    
    return false;
}

void DemandTextureSampler::shutdown()
{
}

void DemandTextureSampler::splitString( const std::string& line, std::vector<std::string>& tokens )
{
    std::string current;
    bool inQuotes = false;
    std::string spaceChars = " (),;{}=:\t\n";
    
    for ( size_t i = 0; i < line.length(); ++i ) 
    {
        char c = line[i];
        if( c == '"' ) 
        {
            inQuotes = !inQuotes;
        }
        else if( !inQuotes && ( spaceChars.find( c ) != std::string::npos ) ) 
        {
            if( !current.empty() ) 
            {
                tokens.push_back( current );
                current.clear();
            }
        }
        else 
        {
            current += c;
        }
    }
    
    if ( !current.empty() )
        tokens.push_back(current);
}

bool DemandTextureSampler::init(  )
{
    if( isInitialized() )
        return true;

    // Initialize CUDA
    OTK_ERROR_CHECK( cuInit( 0 ) );
    OTK_ERROR_CHECK( cudaFree( 0 ) );
    OTK_ERROR_CHECK( cudaSetDevice( 0 ) );
    OTK_ERROR_CHECK( cudaStreamCreate( &m_optixState.stream) );

    // Initialize OptiX context
    {
        CUcontext cuCtx = 0;  // zero means take the current context
        OTK_ERROR_CHECK( optixInit() );
        OptixDeviceContextOptions options = {};
        otk::util::setLogger( options );
        OTK_ERROR_CHECK( optixDeviceContextCreate( cuCtx, &options, &m_optixState.context ) );
    }

    // Initialize DemandLoader
    state.demandLoader.reset( createDemandLoader( options ), demandLoading::destroyDemandLoader );

    // Create OptiX module
    {
        OptixModuleCompileOptions module_compile_options{};
        otk::configModuleCompileOptions( module_compile_options );
        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
        pipeline_compile_options.numPayloadValues      = 2;
        pipeline_compile_options.numAttributeValues    = 2;
        pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

        OTK_ERROR_CHECK_LOG( optixModuleCreate( context, &module_compile_options, &pipeline_compile_options,
                                                draw_solid_colorCudaText(), draw_solid_colorCudaSize, LOG, &LOG_SIZE, &module ) );
    }

    // Set up shader binding table
    {
        CUdeviceptr  raygen_record;
        const size_t raygen_record_size = sizeof( RayGenSbtRecord );
        OTK_ERROR_CHECK( cudaMalloc( reinterpret_cast<void**>( &raygen_record ), raygen_record_size ) );
        RayGenSbtRecord rg_sbt;
        OTK_ERROR_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
        OTK_ERROR_CHECK( cudaMemcpy( reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice ) );
        sbt.raygenRecord = raygen_record;
    }

    return true;
}

bool DemandTextureSampler::loadTexture( std::vector<std::string>& tokens )
{
    init();
    if( tokens.size() < 3 )
    {
        std::cerr << "Error: loadTexture requires 3 arguments" << std::endl;
        return false;
    }

    std::string textureName = tokens[1];
    ImageSourcePtr imageSource;
    demandLoading::TextureDescriptor texDesc = m_textureDesc;

    if( textureName == "checkerboard" )
    {
        if( tokens.size() < 6 )
        {
            std::cerr << "Error: checkerboard requires 6 arguments" << std::endl;
            return false;
        }
        std::string type = tokens[3];
        int width = std::stoi( tokens[4] );
        int height = std::stoi( tokens[5] );
        bool mipmap = tokens.size() >= 7 && tokens[6] == "mipmap";

        if( type == "uchar" )
            imageSource.reset( new imageSource::MultiCheckerImage<uchar>( width, height, 16, mipmap, true ) );
        else if( type == "uchar2" )
            imageSource.reset( new imageSource::MultiCheckerImage<uchar2>( width, height, 16, mipmap, true ) );
        else if( type == "uchar4" )
            imageSource.reset( new imageSource::MultiCheckerImage<uchar4>( width, height, 16, mipmap, true ) );
        else if( type == "half" )
            imageSource.reset( new imageSource::MultiCheckerImage<half>( width, height, 16, mipmap, true ) );
        else if( type == "half2" )
            imageSource.reset( new imageSource::MultiCheckerImage<half2>( width, height, 16, mipmap, true ) );
        else if( type == "half4" )
            imageSource.reset( new imageSource::MultiCheckerImage<half4>( width, height, 16, mipmap, true ) );
        else if( type == "float" )
            imageSource.reset( new imageSource::MultiCheckerImage<float>( width, height, 16, mipmap, true ) );
        else if( type == "float2" )
            imageSource.reset( new imageSource::MultiCheckerImage<float2>( width, height, 16, mipmap, true ) );
        else if( type == "float4" )
            imageSource.reset( new imageSource::MultiCheckerImage<float4>( width, height, 16, mipmap, true ) );
    }
#ifdef INCLUDE_NTC_TEXTURES
    else if( textureName.endsWith( ".ntc" ) )
    {
        imageSource.reset( new NeuralTextureSource( textureName ) );
        texDesc.flags |= CU_TRSF_READ_AS_INTEGER;
    }
#endif
    else
    {
        imageSource = imageSource::createImageSource( textureName );
    }

    if( !imageSource )
    {
        std::cerr << "Error: could not create image source for texture " << textureName << std::endl;
        return false;
    }

    const demandLoading::DemandTexture& texture m_optixState.demandLoader->createTexture( imageSource, texDesc );
    m_textureIdMap[textureName] = texture.getId();

    return true;
}

bool DemandTextureSampler::launch( std::vector<std::string>& tokens )
{
    init();
    state.demandLoader->launchPrepare( state.stream, state.params.demand_texture_context );
    //initLaunchParams( state, numDevices );

    // Copy texture samples to device
    OTK_ERROR_CHECK( cuMemcpy( m_optixState.d_textureSamples, &m_optixState.textureSamples, sizeof( m_optixState.textureSamples ), cudaMemcpyHostToDevice ) );

    // Copy launch params to device
    OTK_ERROR_CHECK( cuMemcpy( m_optixState.d_params, &m_optixState.params, sizeof( m_optixState.params ) ) );

    OTK_ERROR_CHECK( optixLaunch( state.pipeline,  // OptiX pipeline
        state.stream,             // Stream
        state.d_params,           // Launch params
        sizeof( state.params ),   // Param size in bytes
        &state.sbt,               // Shader binding table
        MAX_TEXTURE_SAMPLES,      // Launch width
        1,                        // Launch height
        1                         // launch depth
        ) );

    state.ticket = state.demandLoader->processRequests( state.stream, state.params.demand_texture_context );
    state.ticket.wait();
    numRequestsProcessed += static_cast<unsigned int>( state.ticket.numTasksTotal() );

    // Pull the samples back from the device
    std::vector<TextureSample> rsamples( m_optixState.textureSamples.size() );
    OTK_ERROR_CHECK( cuMemcpy( rsamples.data(), m_optixState.d_textureSamples, sizeof( m_optixState.textureSamples ), cudaMemcpyDeviceToHost ) );

    // Print samples
}

void DemandTextureSampler::printSample( TextureSample& s )
{
    int tfunc = s.textureFunction;
    float4 r = s.result;
    float4 drds = s.dresultds;
    float4 drdt = s.dresultdt;
    std::string funcName = textureFunctionNames[tfunc];

    // Print the texture function and arguments
    if( tfunc == TF_tex2D )
    {
        printf("%s(%s, %0.4f, %0.4f) = ", funcName, idTextureMap[s.textureId], s.u, s.v);
    }
    else if( tfunc == TF_tex2DLod )
    {
        printf("%s(%s, %0.4f, %0.4f, %0.4f) = ", funcName, idTextureMap[s.textureId], s.u, s.v, s.lod);
    }
    else if( tfunc == TF_tex2DGrad || tfunc == TF_tex2DGradUdim || tfunc == TF_tex2DGradUdimBlend 
             || tfunc == TF_tex2DCubic || tfunc == TF_tex2DCubicUdim
             || tfunc == TF_tex2DCubicDerivatives || tfunc == TF_tex2DCubicUdimDerivatives
             || tfunc == TF_ntcTex2DGrad || tfunc == TF_ntcTex2DGradUdim )
    {
        printf("%s(%s, %0.4f, %0.4f, {%0.4f, %0.4f}, {%0.4f, %0.4f}) = ", 
            funcName, idTextureMap[s.textureId], s.u, s.v, s.ddx.x, s.ddx.y, s.ddy.x, s.ddy.y);
    }

    // Print the result
    if( tfunc == TF_tex2D || tfunc == TF_tex2DLod || tfunc == TF_tex2DGrad 
        || tfunc == TF_tex2DGradUdim || tfunc == TF_tex2DGradUdimBlend 
        || tfunc == TF_tex2DCubic || tfunc == TF_tex2DCubicUdim)
    {
        if( s.resident )
            printf("(%0.3f, %0.3f, %0.3f, %0.3f)\n", r.x, r.y, r.z, r.w);
        else
            printf("(NOT RESIDENT)\n");
    }
    else if( tfunc == TF_tex2DCubicDerivatives || tfunc == TF_tex2DCubicUdimDerivatives )
    {
        if( s.resident )
            printf("(%0.3f, %0.3f, %0.3f, %0.3f), (%0.3f, %0.3f, %0.3f, %0.3f)\n", 
                drds.x, drds.y, drds.z, drds.w, drdt.x, drdt.y, drdt.z, drdt.w);
        else
            printf("(NOT RESIDENT)\n");
    }
    else if( tfunc == TF_ntcTex2DGrad || tfunc == TF_ntcTex2DGradUdim )
    {
        if( s.resident )
        {
            printf("[ ")
            for( int i = 0; i < NTC_MLP_OUTPUT_CHANNELS; i++ )
                printf("%0.3f ", s.out[i]);
            printf("]\n");
        }
        else
            printf("(NOT RESIDENT)\n");
    }
}


//
// Main function
//

int main( int argc, char* argv[] )
{
    DemandTextureSampler dtSampler( options );

    if( argc > 1 )
    {
        std::cerr << "\nThe DemandTextureSampler is a test harness to take individual texture samples\n"
            "through the demand loading pipeline over a number of OptiX launches.  Texture\n"
            "loading, sample specification, and OptiX launches are given by commands\n"
            "The results of each launch are printed to standard output.\n"
        dtdSampler.printUsage();
        std::cerr << "\nCommon usage is to write all the commands for a session to a text file\n"
            "and then run the DemandTextureSampler with the file as standard input.\n"
            "\nExample input file:\n\n"
            "    maxRequestedPages 1\n"
            "    loadTexture tex1 \"image.exr\"\n"
            "    loadTexture tex2 checkerboard 1024 768 mipmap\n"
            "    # Code notations like '(){};,=:' are optional\n"
            "    tex2D tex1 0.5 0.5\n"
            "    tex2DGrad(0, 0.35, 0.75, {0.1, 0.1}, {0.1, -0.1}, {0.0, 0.0});\n"
            "    # Texture samples are executed during successive launches.\n"
            "    launch 3\n"
            "\n";
        exit( 0 );
    }

    std::string line;
    while( std::getline(std::cin, line) ) 
    {
        if( !processCommand( line, dtSampler, numTextureSamples, textureSamples ) )
            break;
    }
    dtSampler.shutdown();

    return 0;
}