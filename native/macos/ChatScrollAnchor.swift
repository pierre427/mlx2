import SwiftUI

// Stick-to-bottom state for the chat transcript. The transcript follows new
// content while the reader is at (or near) the bottom; it detaches only when
// the reader deliberately scrolls away, and reattaches once they scroll back.
// A layout pass (lazy-row height estimates settling, a window resize, the
// composer growing) never counts as a deliberate scroll.
struct ChatScrollAttachment {
    static let threshold: CGFloat = 32
    private(set) var followsBottom = true
    private var offset: CGFloat?
    private var contentHeight: CGFloat = 0
    /// - Parameters:
    ///   - offset: vertical content offset.
    ///   - distance: points between the visible bottom edge and the content's end.
    ///   - userScroll: the scroll view reports a user gesture or its momentum.
    mutating func observe(offset: CGFloat, contentHeight: CGFloat, distance: CGFloat, userScroll: Bool) {
        // A scroll wheel may not report a scroll phase; an upward offset move
        // with unchanged content is still the reader's own scroll.
        let movedUp = offset < (self.offset ?? offset) - 0.5 && abs(contentHeight - self.contentHeight) <= 0.5
        if distance <= Self.threshold { followsBottom = true }
        else if userScroll || movedUp { followsBottom = false }
        self.offset = offset; self.contentHeight = contentHeight
    }
    mutating func attach() { followsBottom = true }
}

private struct ChatScrollMetrics: Equatable {
    var offset: CGFloat
    var contentHeight: CGFloat
    var viewport: CGFloat
    var distance: CGFloat
}

struct ChatTranscriptScroll<Content: View>: View {
    let request: UUID
    @ViewBuilder let content: () -> Content
    @State private var attachment = ChatScrollAttachment()
    @State private var position = ScrollPosition(edge: .bottom)
    @State private var phase: ScrollPhase = .idle
    @State private var metrics = ChatScrollMetrics(offset: 0, contentHeight: 0, viewport: 0, distance: 0)

    private func scrollToBottom(animated: Bool) {
        if animated {
            withAnimation(.smooth(duration: 0.28)) { position.scrollTo(edge: .bottom) }
        } else {
            var transaction = Transaction(); transaction.disablesAnimations = true
            withTransaction(transaction) { position.scrollTo(edge: .bottom) }
        }
    }

    var body: some View {
        ScrollView {
            content()
        }
        .scrollPosition($position)
        // While attached, growth keeps the bottom edge pinned natively, in the
        // same layout pass, so streamed text pushes the transcript up with no
        // lag or jump. Detached, the reader's position is left alone.
        .defaultScrollAnchor(.bottom, for: .initialOffset)
        .defaultScrollAnchor(attachment.followsBottom ? .bottom : nil, for: .sizeChanges)
        .onScrollPhaseChange { _, newPhase in
            phase = newPhase
            if newPhase == .idle, attachment.followsBottom, metrics.distance > 1 {
                scrollToBottom(animated: metrics.distance < metrics.viewport * 2)
            }
        }
        .onScrollGeometryChange(for: ChatScrollMetrics.self) { geometry in
            ChatScrollMetrics(offset: geometry.contentOffset.y, contentHeight: geometry.contentSize.height,
                              viewport: geometry.containerSize.height,
                              distance: max(0, geometry.contentSize.height + geometry.contentInsets.bottom - geometry.visibleRect.maxY))
        } action: { _, new in
            metrics = new
            attachment.observe(offset: new.offset, contentHeight: new.contentHeight, distance: new.distance,
                               userScroll: phase == .interacting || phase == .decelerating)
            // Catch-up for anything the size anchor did not absorb (a new
            // message row materialising below the fold). Never restart an
            // animation already in flight: that is what makes scrolling stutter.
            if attachment.followsBottom, new.distance > 1, phase == .idle {
                scrollToBottom(animated: new.distance < new.viewport * 2)
            }
        }
        .onChange(of: request) { _, _ in
            attachment.attach()
            scrollToBottom(animated: metrics.distance < metrics.viewport * 2)
        }
        .frame(minHeight: 0, maxHeight: .infinity)
    }
}
