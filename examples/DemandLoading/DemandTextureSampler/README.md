# demandTextureSampler

The `demandTextureSampler` is a console program intended to help diagnose issues related to demand load textures. It works by allowing a user to load textures, specify texture samples, and track texture reads and demand loading requests through a number of OptiX launches.

### Commands

The demandTextureSampler utility accepts commands to configure the demand loading system, load texture files or synthetic textures, specify texture samples, and perform OptiX launches that take the samples. Results of each launch are printed to the console, including the value returned by each texture call, and demand load requests resulting from each OptiX launch. Commands are divided into 4 categories: general, demand loading options, texture loading, and texture sampling. A complete list is as follows:

```
General Commands.
    help
    launch [<numLaunches>]
    quit

Demand loading options. (Configure the demand loading system options)
    maxRequestedPages <numPages>
    useSparseTextures <true|false>
    useCascadingTextureSizes <true|false>
    coalesceWhiteBlackTiles <true|false>
    coalesceDuplicateImages <true|false>

Texture Loading Commands. (Set address and filtering modes, and load textures)
    addressMode <border|clamp|wrap> <border|clamp|wrap>
    filterMode <point|linear|bicubic|smartbicubic>
    mipmapFilterMode <point|linear>
    maxAnisotropy <1-16>
    conservativeFilter <true|false>
    loadTexture <textureId> <textureName>
    loadTexture <textureId> checkerboard <type> <width> <height> [<mipmap>]
        (type is: uchar, uchar2, uchar4, half, half2, half4, float, float2, or float4)
        
Texture sample commands. (Add texture samples to the next OptiX launch)
    clearSamples (clears all samples for the next launch)
    tex2D <textureId> <u> <v> [<jitter>]
    tex2DLod <textureId> <u> <v> <lod> [<jitter>]
    tex2DGrad <textureId> <u> <v> <ddx> <ddy> [<jitter>]
    tex2DGradUdim <textureId> <u> <v> <ddx> <ddy> [<jitter>]
    tex2DGradUdimBlend <textureId> <u> <v> <ddx> <ddy> [<jitter>]
    tex2DCubic <textureId> <u> <v> <ddx> <ddy> [<jitter>]
    tex2DCubicDerivatives <textureId> <u> <v> <ddx> <ddy> [<jitter>]
    tex2DCubicUdim <textureId> <u> <v> <ddx> <ddy> [<jitter>]
    tex2DCubicUdimDerivatives <textureId> <u> <v> <ddx> <ddy> [<jitter>]
```

### Usage

Since debugging sessions may require many commands and iterations, common usage is to write the commands for a session in a text file and run the demandTextureSampler with the file redirected to standard input. An example input file is shown here:

```
# Example input file. (# and // indicate comment lines)
# Configure the demand loader.
maxRequestedPages 32
useCascadingTextureSizes true

# Set interpolation modes
addressMode border border
filterMode point
mipmapFilterMode point

# Load an image texture.
loadTexture tex1 "image1.exr"

# Load a procedural checkerboard texture.
loadTexture tex2 checkerboard uchar4 1024 768 mipmap 

# Specify some texture samples.
# Code notations like '(){}:;,=' are ignored, but can make samples more readable
# Texture are sampled as float4 without regard to input type
tex2D tex1 0.5 0.5
tex2DGrad(tex2, 0.35, 0.75, {0.1, 0.1}, {0.1, -0.1}, {0.0, 0.0});

# Perform an OptiX launch
launch

# Change interpolation mode for subsequent textures
filterMode smartbicubic
mipmapFilterMode linear

# Load another texture
loadTexture tex3 "image2.exr"

# Add another texture samples for the next launches
tex2DCubic(tex3, 0.1, 0.23, {0.01, 0.11}, {-0.11, 0.01});

# Launch 3 more times
launch 3

quit
```
