---
title: "C# Hello World: from source to assembly"
---

How much is hidden behind a single line of C#?

```csharp
System.Console.WriteLine("Hello, world!");
```

A single line of C# passes through several layers before text appears in the terminal: compiler output, the .NET runtime, native instructions, and operating-system I/O. Follow those layers from the source code down to the CPU and back out to standard output.

## The route down

1. [[Programming Rabbit Hole/CSharp Hello World/01 Source and build|Source and build]] — establish the smallest reproducible program and inspect its build output.
2. [[Programming Rabbit Hole/CSharp Hello World/02 Inside the assembly|Inside the .NET assembly]] — inspect metadata, the entry point, and intermediate language.
3. [[Programming Rabbit Hole/CSharp Hello World/03 Runtime and JIT|Runtime and JIT]] — follow startup and the transition from IL to native code.
4. [[Programming Rabbit Hole/CSharp Hello World/04 Native instructions|Native instructions]] — read the disassembly and connect instructions back to the program.
5. [[Programming Rabbit Hole/CSharp Hello World/05 Reaching the terminal|Reaching the terminal]] — follow the output path beyond the call to `Console.WriteLine`.

## Reading the output

Compiler output and native instructions depend on the SDK and runtime versions, operating system, CPU architecture, and build settings. A disassembly describes a particular build and execution environment.

Beyond the output call, the path continues through operating-system I/O and terminal rendering. Executable loading, virtual memory, instruction encoding, and CPU execution each open another layer to explore.
