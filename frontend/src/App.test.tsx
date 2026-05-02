import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App from "./App";

describe("App", () => {
  it("renders the empty state without crashing", () => {
    render(<App />);
    expect(screen.getByText(/Route retired cells by value/i)).toBeInTheDocument();
  });
});
