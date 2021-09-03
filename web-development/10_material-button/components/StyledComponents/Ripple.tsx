import styled from "styled-components";

interface RippleProps {
    color?: string;
}

export default styled.span<RippleProps>`
    background-color: ${props => props.color};
    opacity: 0.3;
    border-radius: 50%;
    position: absolute;
    transform: scale(0);
    pointer-events: none;
`