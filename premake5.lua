include "Dependencies.lua"

workspace "PreisachModel"
	configurations { "Debug", "Release" }
	targetdir "build"
	startproject "PreisachModel"
	conformancemode "On"

	language "C++"
	cppdialect "C++20"
	staticruntime "Off"

	configurations
	{ 
		"Debug", 
		"Release",
	}

	flags
	{
		"MultiProcessorCompile"
	}

	-- NOTE(Peter): Don't remove this. Please never use Annex K functions ("secure", e.g _s) functions.
	defines {
		"_CRT_SECURE_NO_WARNINGS",
		"NOMINMAX",
		"_SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING",
		"TRACY_ENABLE",
		"TRACY_ON_DEMAND",
		"TRACY_CALLSTACK=10",
		"NUMCPP_NO_USE_BOOST",
	}

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