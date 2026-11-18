// @file
// @author Vlayd Detect <gingema62@gmail.com>
// [GitHub Repository](https://github.com/VlaydDetect/MUTCpp)
// License
// Copyright (c) 2024 Vlayd Detect.All rights reserved.

#pragma once

#define AL_ENABLE_PROFILING AL_DEBUG

#if AL_ENABLE_PROFILING
#include <tracy/Tracy.hpp>
#endif

#if AL_ENABLE_PROFILING
#define AL_PROFILE_MARK_FRAME			FrameMark;
// NOTE(Peter): Use AL_PROFILE_FUNC ONLY at the top of a function
//				Use AL_PROFILE_SCOPE / AL_PROFILE_SCOPE_DYNAMIC for an inner scope
#define AL_PROFILE_FUNC(...)			ZoneScoped##__VA_OPT__(N(__VA_ARGS__))
#define AL_PROFILE_SCOPE(...)			AL_PROFILE_FUNC(__VA_ARGS__)
#define AL_PROFILE_SCOPE_DYNAMIC(NAME)  ZoneScoped; ZoneName(NAME, strlen(NAME))
#define AL_PROFILE_THREAD(...)          tracy::SetThreadName(__VA_ARGS__)
#else
#define AL_PROFILE_FUNC(...)
#define AL_PROFILE_SCOPE(...)
#define AL_PROFILE_SCOPE_DYNAMIC(NAME)
#define AL_PROFILE_THREAD(...)
#endif
