import React from 'react';
import { NavigationBar } from './navigation.bar';
import { HeroSection } from './hero.section';
import { Features } from './features';
import { ChooseUsSection } from './choose.section';
import { GetStarted } from './get.started';
import { Footer } from './footer';

export default function LandingPage() {
  return (
    <div className="flex flex-col min-h-screen bg-[#1b1724] text-white">
      <NavigationBar />
      <HeroSection />
      <Features />
      <ChooseUsSection />
      <GetStarted />
      <Footer />
    </div>
  );
}
