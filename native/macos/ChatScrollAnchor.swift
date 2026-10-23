import SwiftUI

// Geometry belongs to the transcript's named coordinate space. Never reach for
// an enclosing AppKit scroll view: SwiftUI can place the entire window inside one.
struct ChatScrollAttachment {
    private(set) var followsBottom = true
    private var documentHeight: CGFloat = 0
    private var viewportHeight: CGFloat = 0
    mutating func observe(height: CGFloat, viewport: CGFloat, distance: CGFloat, userScroll: Bool? = nil) {
        let resized = abs(height - documentHeight) > 0.5 || abs(viewport - viewportHeight) > 0.5
        if distance <= 32 { followsBottom = true }
        else if userScroll == true || (userScroll == nil && !resized) { followsBottom = false }
        documentHeight = height; viewportHeight = viewport
    }
    mutating func attach() { followsBottom = true }
}

private struct ChatTranscriptFrame: PreferenceKey {
    static var defaultValue: CGRect { .zero }
    static func reduce(value: inout CGRect, nextValue: () -> CGRect) { value = nextValue() }
}

private struct ChatUserScrollPhase: ViewModifier {
    let changed: (Bool) -> Void
    @ViewBuilder func body(content: Content) -> some View {
        if #available(macOS 15.0, *) {
            content.onScrollPhaseChange { _, phase in changed(phase == .interacting || phase == .decelerating) }
        } else { content }
    }
}

struct ChatTranscriptScroll<Content: View>: View {
    let request: UUID
    @ViewBuilder let content: () -> Content
    @State private var attachment = ChatScrollAttachment()
    @State private var space = UUID()
    @State private var end = UUID()
    @State private var pending = false
    @State private var userScrolling = false
    private var scrollIntent: Bool? {
        if #available(macOS 15.0, *) { return userScrolling }
        return nil
    }
    private func follow(_ proxy: ScrollViewProxy) {
        guard attachment.followsBottom, !pending else { return }
        pending = true
        Task { @MainActor in
            await Task.yield()
            pending = false
            guard attachment.followsBottom else { return }
            proxy.scrollTo(end, anchor: .bottom)
        }
    }
    var body: some View {
        GeometryReader { viewport in
            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 0) {
                        content()
                        Color.clear.frame(height: 1).id(end)
                    }
                    .background(GeometryReader { geometry in
                        Color.clear.preference(key: ChatTranscriptFrame.self, value: geometry.frame(in: .named(space)))
                    })
                }
                .coordinateSpace(name: space)
                .modifier(ChatUserScrollPhase { userScrolling = $0 })
                .onPreferenceChange(ChatTranscriptFrame.self) { frame in
                    guard frame.height > 0 else { return }
                    attachment.observe(height: frame.height, viewport: viewport.size.height,
                                       distance: frame.maxY - viewport.size.height, userScroll: scrollIntent)
                    follow(proxy)
                }
                .onChange(of: request) { _, _ in attachment.attach(); follow(proxy) }
            }
        }.frame(minHeight: 0, maxHeight: .infinity)
    }
}
