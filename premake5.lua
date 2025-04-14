include "Dependencies.lua"

workspace "PreisachModel"
	configurations { "Debug", "Release" }
	startproject "PreisachModel"
	conformancemode "On"

	language "C++"
    --toolset "msc-clangcl"
    --llvmversion "19"
    --llvmdir "D:/Programming/SDKs/LLVM"
    cppdialect "C++23"
	staticruntime "Off"

	flags
	{
		"MultiProcessorCompile"
	}

	-- NOTE(Peter): Don't remove this. Please never use Annex K functions ("secure", e.g _s) functions.
	defines {
		"_CRT_SECURE_NO_WARNINGS",
		"NOMINMAX",
		"_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING",
		--"MUTCPP_USE_MULTITHREAD",
		
		"TRACY_ENABLE",
        "TRACY_ON_DEMAND",
        "TRACY_CALLSTACK=10",
	}
	
	filter "action:vs*"
        linkoptions {"/ignore:4099"} -- NOTE(Peter): Disable no PDB found warning
        disablewarnings {"4068"} -- Disable "Unknown #pragma mark warning"

    filter "language:C++ or language:C"
        architecture "x86_64"

    filter "configurations:Debug"
        optimize "Off"
        symbols "On"

    filter {"system:windows"}
        flags {"NoRuntimeChecks", "NoIncrementalLink"}

    filter "configurations:Release"
        optimize "On"
        symbols "Default"

    filter "system:windows"
        buildoptions {"/EHsc", "/Zc:preprocessor", "/Zc:__cplusplus"}


outputdir = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

include "PreisachModel"
include "PreisachModel/vendor/tracy"
