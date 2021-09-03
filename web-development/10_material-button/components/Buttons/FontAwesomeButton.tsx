import React from "react";
import styled from "styled-components";
import * as FontAwesome from "react-icons/fa";
import { BoxShadowStyle, RippleTrigger, halfTheSize, getRGBA } from "../../functions";
import Ripple from "../StyledComponents/Ripple";

interface Props {
    fontawesome: string;
    href?: string;
    color?: string;
    backgroundColor?: string;
    opacity?: string;
    size?: string;
    margin?: string;
    border?: string;
    rippleColor?: string;
}

const FontAwesomeButton:React.FC<Props> = ({ fontawesome, href = "#", color = "#ffffff", backgroundColor = "#9c27b0", opacity = "1", size = "40px", margin = "1rem", border="2px", rippleColor = "#ffffff" }) => {
    const rippleRef = React.createRef() as React.RefObject<HTMLSpanElement>;
    return (
        <ButtonContainer href={href} color={color} backgroundColor={backgroundColor} opacity={opacity} size={size} margin={margin} border={border} onClick={e => RippleTrigger(e, rippleRef.current as HTMLSpanElement)}>
            {React.createElement(FontAwesome[fontawesome])}
            <Ripple ref={rippleRef} color={rippleColor}/>
        </ButtonContainer>
    ) 
}

export default FontAwesomeButton;

interface ButtonProps {
    color: string;
    backgroundColor: string;
    opacity: string;
    size: string;
    margin: string;
    border: string;
}

const ButtonContainer = styled.a<ButtonProps>`
    text-decoration: none;
    color: ${props => props.color};
    font-size: ${props => halfTheSize(props.size)};
    background-color: ${props => getRGBA(props.backgroundColor, props.opacity as any)};
    box-shadow: ${props => BoxShadowStyle(false, props.backgroundColor)};
    border-radius: ${props => props.border};
    width: ${props => props.size};
    height: ${props => props.size};
    margin: ${props => props.margin};
    display: inline-flex;
    justify-content: center;
    align-items: center;
    position: relative;
    transition: ${props => props.theme.primaryTransition};
    overflow: hidden;

    &:hover,
    &:focus {
        box-shadow: ${props => BoxShadowStyle(true, props.backgroundColor)};
    }
`