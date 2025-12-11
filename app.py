"use client";
import { Home, MessageSquare, BarChart3, Database, FileUp, BookOpen } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

// The complete navigation configuration
const navItems = [
  { name: "Home", href: "/", icon: Home },
  { name: "Prompt Agent", href: "/agent", icon: MessageSquare },
  { name: "Analytics", href: "/dashboard", icon: BarChart3 },
  { name: "Explorer", href: "/explorer", icon: Database },
  { name: "File Runner", href: "/runner", icon: FileUp },
  { name: "Documentation", href: "/docs", icon: BookOpen },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-64 border-r border-white/10 bg-black/40 backdrop-blur-xl">
      <div className="flex h-full flex-col p-6">
        
        {/* LOGO AREA (Text Only & Clickable) */}
        <Link 
          href="/" 
          className="mb-10 block hover:opacity-80 transition-opacity cursor-pointer"
        >
          <h1 className="text-2xl font-bold tracking-tight text-white pl-2">
            SolarOS
          </h1>
        </Link>

        {/* NAVIGATION LINKS */}
        <nav className="flex flex-col gap-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? "bg-white/10 text-white shadow-[0_0_10px_rgba(255,255,255,0.05)] border border-white/5"
                    : "text-neutral-400 hover:bg-white/5 hover:text-white"
                }`}
              >
                <item.icon className="h-5 w-5" />
                {item.name}
              </Link>
            );
          })}
        </nav>
        
        {/* USER PROFILE FOOTER (Linked to Account Page) */}
        <div className="mt-auto pt-6 border-t border-white/10">
           <Link href="/account">
               <div className="flex items-center gap-3 group cursor-pointer px-2 py-2 rounded-xl hover:bg-white/5 transition-colors">
                  <div className="h-8 w-8 rounded-full bg-gradient-to-tr from-blue-600 to-blue-400 border border-white/20 shadow-lg" />
                  <div>
                    <div className="text-sm font-medium text-white group-hover:text-blue-400 transition-colors">Admin User</div>
                    <div className="text-xs text-neutral-500">Pro Plan • Active</div>
                  </div>
               </div>
           </Link>
        </div>
      </div>
    </aside>
  );
}
