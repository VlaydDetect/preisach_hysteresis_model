include "./ordered_pairs.lua"

function firstToUpper(str)
    return (str:gsub("^%l", string.upper))
end

Dependencies = {
    NumCpp = {
        IncludeDir = "%{wks.location}/PreisachModel/vendor/NumCpp/include"
    },
    enumerate = {
        IncludeDir = "%{wks.location}/PreisachModel/vendor/hipony/"
    }
    --Boost = {
    --    LibDir = "%{wks.location}/PreisachModel/vendor/boost/bin",
    --    IncludeDir = "%{wks.location}/PreisachModel/vendor/boost/include"
    --
    --}
}

-- NOTE(Peter): Probably don't touch these functions unless you know what you're doing (or just ask me if you need help extending them)

function LinkDependency(table, is_debug, target)

    -- Setup library directory
    if table.LibDir ~= nil then
        libdirs {table.LibDir}
    end

    -- Try linking
    local libraryName = nil
    if table.LibName ~= nil then
        libraryName = table.LibName
    end

    if table.DebugLibName ~= nil and is_debug and target == "Windows" then
        libraryName = table.DebugLibName
    end

    if libraryName ~= nil then
        links {libraryName}
        return true
    end

    return false
end

function AddDependencyIncludes(table)
    if table.IncludeDir ~= nil then
        externalincludedirs {table.IncludeDir}
    end
end

function ProcessDependencies(config_name)
    local target = firstToUpper(os.target())

    for key, libraryData in orderedPairs(Dependencies) do

        -- Always match config_name if no Configurations list is specified
        local matchesConfiguration = true

        if config_name ~= nil and libraryData.Configurations ~= nil then
            matchesConfiguration = string.find(libraryData.Configurations, config_name)
        end

        local isDebug = config_name == "Debug"

        if matchesConfiguration then
            local continueLink = true

            -- Process Platform Scope
            if libraryData[target] ~= nil then
                continueLink = not LinkDependency(libraryData[target], isDebug, target)
                AddDependencyIncludes(libraryData[target])
            end

            -- Process Global Scope
            if continueLink then
                LinkDependency(libraryData, isDebug, target)
            end

            AddDependencyIncludes(libraryData)
        end

    end
end

function IncludeDependencies()
    local target = firstToUpper(os.target())

    for key, libraryData in orderedPairs(Dependencies) do

        -- Process Global Scope
        AddDependencyIncludes(libraryData)

        -- Process Platform Scope
        if libraryData[target] ~= nil then
            AddDependencyIncludes(libraryData[target])
        end

    end
end

