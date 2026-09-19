import Foundation

let info = ProcessInfo.processInfo
print("{\"thermal_state\":\(info.thermalState.rawValue)}")
