import "./App.css";
import GraphTrackerV2 from "./Components/DashboardV2/GraphTrackerV2";
import LandingPage from "./Components/LandingPage";
import { Login } from "./Components/Login/login";
import { SignUp } from "./Components/SignUp/signup";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import { Toaster } from 'react-hot-toast';

function App() {
  return (
    <>
      <Toaster />
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/dashboard" element={<GraphTrackerV2 />} />
          
          {/* Fallback to redirect unknown paths to the landing page */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
