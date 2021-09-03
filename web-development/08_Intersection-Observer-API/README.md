# Introduction
The Intersection Observer API provides a way to **asynchronously** observe changes in the intersection of a target element with an ancestor element or with a top-level document's viewport.

# Examples
Lazy-loading, Infinite Scroll, Scroll Magic
- [Single Observer](https://rayyamhk.github.io/Self-learning/web-development/08_Intersection-Observer-API/Demonstrations/Single%20Observer%20With%20Explainations/)
- [Multiple Observers](https://rayyamhk.github.io/Self-learning/web-development/08_Intersection-Observer-API/Demonstrations/Multiple%20Observers/)
- [Infinite Scroll (Shrink the screen if not working)](https://rayyamhk.github.io/Self-learning/web-development/08_Intersection-Observer-API/Infinite%20Scroll)
- [Lazy Load](https://rayyamhk.github.io/Self-learning/web-development/08_Intersection-Observer-API/Lazy%20Loading/)

# Configuration
**root**: The element used for intersection checking. It should be the ancestor of the targets.<br>
**rootMargin**: A string with values in the same format as for a CSS margin value. This creates a margin around the root element. It defaults to '0px'.<br>
**threshold**: An array of number values between 0 and 1. The values correspond to the ratio of visibility of the element. If you provide multiple values, the intersection callback will be called when each specified threshold value is reached. It defaults to [0].

# Remarks
- If the targets don't need to be observed anymore, you can unobserve them by observer.unobserve(target), e.g. Lazy-loading <br>
- You can declare multiple observers to handle different targets <br>
- For each target, the callback will be normally executed twice, i.e. entering and leaving <br>
- Since Intersection Observer API is asynchronous, it is not simultaneous executed and the callback involved should not be too "heavy"

# Browser Compatibility
https://caniuse.com/#search=intersectionobserver<br>
Interaction Observer Polyfill: https://github.com/w3c/IntersectionObserver/tree/master/polyfill

# References
- https://developer.mozilla.org/en-US/docs/Web/API/Intersection_Observer_API <br>
- https://alligator.io/js/intersection-observer/ <br>
- **rootMargin**: https://www.youtube.com/watch?v=T8EYosX4NOo 9:50
