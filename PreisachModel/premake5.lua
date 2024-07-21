project "PreisachModel"
	kind "ConsoleApp"

	targetdir ("%{wks.location}/bin/" .. outputdir .. "/%{prj.name}")
	objdir ("%{wks.location}/intermediate/" .. outputdir .. "/%{prj.name}")

	files
	{
		"source/**.h",
		"source/**.c",
		"source/**.hpp",
		"source/**.cpp",

        "vendor/NumCpp/include/**.hpp",
        "vendor/hipony/enumerate.hpp"
	}

	includedirs
	{
		"source/",
		"vendor/",
	}

    --links {
    --    Dependencies.Boost.LibDir,
    --}

	IncludeDependencies()

	filter "configurations:Debug"
		symbols "On"
		defines { "_DEBUG", "ACL_ON_ASSERT_ABORT", }
	
	filter "configurations:Release"
		optimize "On"
		vectorextensions "AVX2"
		isaextensions { "BMI", "POPCNT", "LZCNT", "F16C" }
		defines { "NDEBUG", }
