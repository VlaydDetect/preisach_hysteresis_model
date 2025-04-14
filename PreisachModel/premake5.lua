project "PreisachModel"
	kind "ConsoleApp"
	
	debuggertype "Mixed"

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/intermediate/" .. outputdir .. "/%{prj.name}")

	files
	{
		"source/**.h",
		"source/**.c",
		"source/**.hpp",
		"source/**.cpp",

		"vendor/MUTCpp/include/**.hpp",
		"vendor/MUTCpp/include/**.cpp",
		"vendor/MUTCpp/vendor/**.hpp",
		"vendor/hep-mc/include/**.hpp",
	}

	includedirs
	{
		"source/",
		"vendor/",
		"vendor/MUTCpp/include/",
		"vendor/MUTCpp/include/MUTCpp",
		"vendor/hep-mc/include/",
	}
	
	IncludeDependencies()
	
	defines {
        "_HAS_CXX17",
        "_HAS_CXX20",
        "_HAS_CXX23",
    }

    links {
        "Ws2_32"
    }

	filter "configurations:Debug"
		symbols "On"
		defines { "_DEBUG", "ACL_ON_ASSERT_ABORT", }
		ProcessDependencies("Debug")
	
	filter "configurations:Release"
        optimize "On"
        vectorextensions "AVX2"
        isaextensions { "BMI", "POPCNT", "LZCNT", "F16C" }
        
        ProcessDependencies("Release")
