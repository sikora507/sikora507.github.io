---
title: "1. Source and build"
---

What does a C# build actually produce? Let's examine the files in `bin/Debug/net10.0` after compiling Hello World on macOS.

The SDK used here is:

```sh
dotnet --list-sdks
```

```text
10.0.400 [/usr/local/share/dotnet/sdk]
```

Create the application:

```sh
dotnet new console -n Hello
cd Hello
cat Program.cs
```

```csharp
Console.WriteLine("Hello, World!");
```

```sh
dotnet build
```

## Inside bin/Debug/net10.0

The default build puts the output in `bin/Debug/net10.0`: `Debug` is the build configuration, and `net10.0` is the target framework. Enter that directory:

```sh
cd bin/Debug/net10.0
du -sh *
```

```text
124K    Hello
4.0K    Hello.deps.json
8.0K    Hello.dll
 12K    Hello.pdb
4.0K    Hello.runtimeconfig.json
```

`du` reports disk usage. `-s` prints one summary per entry, `-h` uses human-readable units, and the shell expands `*` to the non-hidden entries in the current directory. Each line shows the disk space used followed by the file name.

These values are from this build on macOS. They measure allocated disk space, rather than exact file lengths in bytes, and can vary with the build and filesystem.

One line of source has produced five files:

| File                       | Role                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------------ |
| `Hello`                    | The native macOS launcher, called the **apphost**, which starts the .NET application.            |
| `Hello.dll`                | The compiled .NET assembly containing our program's **intermediate language (IL)** and metadata. |
| `Hello.pdb`                | Debugging information that connects compiled code to source locations.                           |
| `Hello.deps.json`          | Dependency information used when loading the application.                                        |
| `Hello.runtimeconfig.json` | Runtime requirements and settings.                                                               |

These are the standard [build output files](https://learn.microsoft.com/en-us/dotnet/core/tools/dotnet-build). The file to inspect for our compiled C# code is **`Hello.dll`**.

## Is this already machine code?

For this ordinary build, the C# compiler translates the source into IL. The runtime's JIT compiler translates methods into native instructions during execution. `Hello.dll` therefore contains an intermediate representation of our statement; the native launcher `Hello` serves a different purpose.

On the Apple Silicon Mac used here, inspect the launcher with:

```sh
file Hello
```

```text
Hello: Mach-O 64-bit executable arm64
```

That identifies the launcher's architecture. It doesn't show the native instructions the JIT will generate for our C# statement.

## Run the compiled program

```sh
./Hello
```

```text
Hello, World!
```

We can also start the same assembly through the `dotnet` host:

```sh
dotnet Hello.dll
```

Both are supported ways to [run a framework-dependent application](https://learn.microsoft.com/en-us/dotnet/core/deploying/).

Next, open `Hello.dll` and see what IL the compiler produced for `Console.WriteLine`.

## What is the difference between Hello and Hello.dll?

**`Hello` is the native launcher; `Hello.dll` contains the compiled C# program.**

`Hello` is a macOS executable called the **apphost**. In this build, it contains ARM64 machine code that starts the .NET hosting process, locates the required runtime, and arranges for `Hello.dll` to run. The SDK supplies this launcher and configures it for the application.

`Hello.dll` is a **managed .NET assembly** containing the IL and metadata produced from our C# source. Its `.dll` extension does not make it a Windows-only file or mean it is merely a library: this assembly contains the application's entry point. The .NET runtime loads it, and the JIT compiles its methods into native instructions as needed.

When we run `./Hello`, the apphost starts the application. When we run `dotnet Hello.dll`, the `dotnet` host starts it instead. Both routes execute the same managed program and require a compatible installed .NET runtime for this framework-dependent build. See [.NET application hosting](https://learn.microsoft.com/en-us/dotnet/core/deploying/).

To follow what happened to `Console.WriteLine("Hello, World!")`, we inspect **`Hello.dll`** next.

## Does Hello need Hello.dll to run?

Yes. We can test this by removing `Hello.dll` from the build directory and running `./Hello` again. The launcher reports that the application to execute does not exist and prints the path where it expected to find `Hello.dll`.

```sh
./Hello
```

```text
The application to execute does not exist: '/path/to/Hello/bin/Debug/net10.0/Hello.dll'.
```

The path above is a placeholder; the error shows the full path to the missing assembly on your machine.

This shows that the launcher does not contain our compiled C# program. It expects the separate managed assembly to be present. Running `dotnet build` again restores the missing build output.

The dependency goes one way: with `Hello.dll` and its supporting configuration files present, we can run `dotnet Hello.dll` without the `Hello` launcher. Both launch methods still require a compatible installed .NET runtime for this build.

[[Programming Rabbit Hole/CSharp Hello World/index|Overview]] · Next: [[Programming Rabbit Hole/CSharp Hello World/02 Inside the assembly|Inside the .NET assembly]]
