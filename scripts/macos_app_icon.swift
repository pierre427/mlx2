import AppKit

// Original native icon artwork, rendered at each iconset resolution.
let directory = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let pixels = size * scale
        let image = NSImage(size: NSSize(width: pixels, height: pixels))
        image.lockFocus()
        let inset = CGFloat(pixels) * 0.06
        let tile = NSBezierPath(roundedRect: NSRect(x: inset, y: inset, width: CGFloat(pixels) - 2 * inset,
                                                   height: CGFloat(pixels) - 2 * inset),
                                xRadius: CGFloat(pixels) * 0.21, yRadius: CGFloat(pixels) * 0.21)
        NSGradient(starting: NSColor(srgbRed: 0.15, green: 0.48, blue: 0.96, alpha: 1),
                   ending: NSColor(srgbRed: 0.13, green: 0.25, blue: 0.70, alpha: 1))!.draw(in: tile, angle: -90)
        if let cube = NSImage(systemSymbolName: "cube.transparent", accessibilityDescription: "MLX2")?
            .withSymbolConfiguration(NSImage.SymbolConfiguration(pointSize: CGFloat(pixels) * 0.48, weight: .medium)
                .applying(.init(paletteColors: [.white]))) {
            cube.draw(in: NSRect(x: CGFloat(pixels) * 0.22, y: CGFloat(pixels) * 0.25,
                                width: CGFloat(pixels) * 0.56, height: CGFloat(pixels) * 0.56))
        }
        if pixels >= 128 {
            let attributes: [NSAttributedString.Key: Any] = [.font: NSFont.systemFont(ofSize: CGFloat(pixels) * 0.095, weight: .bold),
                                                           .foregroundColor: NSColor.white]
            let text = "MLX2" as NSString
            let bounds = text.size(withAttributes: attributes)
            text.draw(at: NSPoint(x: (CGFloat(pixels) - bounds.width) / 2, y: CGFloat(pixels) * 0.16), withAttributes: attributes)
        }
        image.unlockFocus()
        let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
        let data = bitmap.representation(using: .png, properties: [:])!
        let name = "icon_\(size)x\(size)\(scale == 2 ? "@2x" : "").png"
        try data.write(to: directory.appendingPathComponent(name))
    }
}
