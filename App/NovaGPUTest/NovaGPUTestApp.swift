// NovaGPUTestApp.swift — "NovaGPU Test": proves the NovaGPU over
// NovaBus. Three pages: a live rotating 3D scene fed only through the
// bus, a self-test suite, and the NovaCPU compatibility proof (the
// real NovaCPU core runs a Nova assembly driver that renders through
// the bus and services the GPU's asset request itself).

import SwiftUI

@main
struct NovaGPUTestApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

struct ContentView: View {
    var body: some View {
        TabView {
            LiveSceneView()
                .tabItem { Label("Live Scene", systemImage: "cube.transparent") }
            VectorsView()
                .tabItem { Label("Vectors", systemImage: "scribble.variable") }
            SelfTestView()
                .tabItem { Label("Self-Tests", systemImage: "checklist") }
            CPUProofView()
                .tabItem { Label("CPU Proof", systemImage: "cpu") }
        }
    }
}
